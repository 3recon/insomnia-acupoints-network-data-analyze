from __future__ import annotations

import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
NODES_INPUT = DATA_DIR / "all_nodes.csv"
EDGES_INPUT = DATA_DIR / "all_edges.csv"
GRAPHML_OUTPUT = OUTPUT_DIR / "all_network.graphml"
SVG_OUTPUT = OUTPUT_DIR / "all_network_top30.svg"
FULL_SVG_OUTPUT = OUTPUT_DIR / "all_network_full.svg"
SUMMARY_OUTPUT = OUTPUT_DIR / "network_summary.txt"

SVG_WIDTH = 1600
SVG_HEIGHT = 1200
TOP_NODE_COUNT = 30
RANDOM_SEED = 42


@dataclass
class Node:
    name: str
    system: str
    frequency: int


@dataclass
class Edge:
    source: str
    target: str
    weight: int


def read_nodes(path: Path) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node = Node(
                name=row["node"],
                system=row["system"],
                frequency=int(row["frequency"]),
            )
            nodes[node.name] = node
    return nodes


def read_edges(path: Path) -> list[Edge]:
    edges: list[Edge] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edges.append(
                Edge(
                    source=row["source"],
                    target=row["target"],
                    weight=int(row["weight"]),
                )
            )
    return edges


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_graphml(nodes: dict[str, Node], edges: list[Edge], output_path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="system" for="node" attr.name="system" attr.type="string"/>',
        '  <key id="frequency" for="node" attr.name="frequency" attr.type="int"/>',
        '  <key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
        '  <graph id="G" edgedefault="undirected">',
    ]

    for node in sorted(nodes.values(), key=lambda item: item.name):
        lines.extend(
            [
                f'    <node id="{xml_escape(node.name)}">',
                f'      <data key="system">{xml_escape(node.system)}</data>',
                f'      <data key="frequency">{node.frequency}</data>',
                "    </node>",
            ]
        )

    for index, edge in enumerate(edges):
        lines.extend(
            [
                f'    <edge id="e{index}" source="{xml_escape(edge.source)}" target="{xml_escape(edge.target)}">',
                f'      <data key="weight">{edge.weight}</data>',
                "    </edge>",
            ]
        )

    lines.extend(["  </graph>", "</graphml>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def top_nodes(nodes: dict[str, Node], count: int) -> list[Node]:
    ranked = sorted(nodes.values(), key=lambda item: (-item.frequency, item.name))
    return ranked[:count]


def filter_edges_for_nodes(edges: Iterable[Edge], node_names: set[str]) -> list[Edge]:
    return [
        edge
        for edge in edges
        if edge.source in node_names and edge.target in node_names
    ]


def spring_layout(
    node_names: list[str],
    edges: list[Edge],
    width: int,
    height: int,
    iterations: int = 200,
) -> dict[str, tuple[float, float]]:
    random.seed(RANDOM_SEED)
    area = width * height
    k = math.sqrt(area / max(len(node_names), 1))
    positions = {
        name: (
            random.uniform(width * 0.15, width * 0.85),
            random.uniform(height * 0.15, height * 0.85),
        )
        for name in node_names
    }

    for step in range(iterations):
        temperature = max(width, height) * 0.12 * (1 - step / iterations)
        displacements = {name: [0.0, 0.0] for name in node_names}

        for index, source in enumerate(node_names):
            x1, y1 = positions[source]
            for target in node_names[index + 1 :]:
                x2, y2 = positions[target]
                dx = x1 - x2
                dy = y1 - y2
                distance = math.hypot(dx, dy) + 0.01
                force = (k * k) / distance
                nx = dx / distance
                ny = dy / distance
                displacements[source][0] += nx * force
                displacements[source][1] += ny * force
                displacements[target][0] -= nx * force
                displacements[target][1] -= ny * force

        for edge in edges:
            x1, y1 = positions[edge.source]
            x2, y2 = positions[edge.target]
            dx = x1 - x2
            dy = y1 - y2
            distance = math.hypot(dx, dy) + 0.01
            force = (distance * distance) / k
            weight_scale = 1 + math.log(edge.weight + 1)
            nx = dx / distance
            ny = dy / distance
            displacements[edge.source][0] -= nx * force * weight_scale
            displacements[edge.source][1] -= ny * force * weight_scale
            displacements[edge.target][0] += nx * force * weight_scale
            displacements[edge.target][1] += ny * force * weight_scale

        for name in node_names:
            dx, dy = displacements[name]
            displacement = math.hypot(dx, dy)
            if displacement > 0:
                limited = min(displacement, temperature)
                x, y = positions[name]
                x += dx / displacement * limited
                y += dy / displacement * limited
                positions[name] = (
                    min(width * 0.92, max(width * 0.08, x)),
                    min(height * 0.92, max(height * 0.08, y)),
                )

    return positions


def node_radius(frequency: int, max_frequency: int) -> float:
    if max_frequency <= 0:
        return 10.0
    return 10.0 + 22.0 * math.sqrt(frequency / max_frequency)


def edge_stroke(weight: int, max_weight: int) -> float:
    if max_weight <= 0:
        return 1.0
    return 1.2 + 5.0 * (weight / max_weight)


def write_svg(nodes: list[Node], edges: list[Edge], output_path: Path) -> None:
    names = [node.name for node in nodes]
    positions = spring_layout(names, edges, SVG_WIDTH, SVG_HEIGHT)
    max_frequency = max(node.frequency for node in nodes)
    max_weight = max(edge.weight for edge in edges) if edges else 1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "  <style>",
        "    .title { font: 700 34px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .subtitle { font: 500 18px 'Malgun Gothic', sans-serif; fill: #4b5563; }",
        "    .label { font: 600 14px 'Malgun Gothic', sans-serif; fill: #111827; }",
        "    .legend { font: 500 16px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
        '  <text class="title" x="60" y="70">\ubd88\uba74\uc99d \uacbd\ud608 \uacf5\ucd9c\ud604 \ub124\ud2b8\uc6cc\ud06c</text>',
        '  <text class="subtitle" x="60" y="105">\uc0c1\uc704 30\uac1c \ub178\ub4dc \uae30\uc900 \uac00\uc911\uce58 \ub124\ud2b8\uc6cc\ud06c \uc2dc\uac01\ud654</text>',
    ]

    for edge in edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        opacity = 0.18 + 0.42 * (edge.weight / max_weight)
        stroke_width = edge_stroke(edge.weight, max_weight)
        lines.append(
            f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#94a3b8" stroke-opacity="{opacity:.2f}" stroke-width="{stroke_width:.2f}"/>'
        )

    for node in nodes:
        x, y = positions[node.name]
        radius = node_radius(node.frequency, max_frequency)
        fill = "#dc2626" if node.system == "ear" else "#2563eb"
        lines.extend(
            [
                f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" fill-opacity="0.85" stroke="#ffffff" stroke-width="2.5"/>',
                f'  <text class="label" x="{x:.2f}" y="{y + radius + 18:.2f}" text-anchor="middle">{xml_escape(node.name)}</text>',
            ]
        )

    lines.extend(
        [
            '  <circle cx="1280" cy="90" r="12" fill="#2563eb" fill-opacity="0.85"/>',
            '  <text class="legend" x="1305" y="96">\uccb4\uce68 \uacbd\ud608</text>',
            '  <circle cx="1280" cy="125" r="12" fill="#dc2626" fill-opacity="0.85"/>',
            '  <text class="legend" x="1305" y="131">\uc774\uce68 \uacbd\ud608</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_full_svg(nodes: dict[str, Node], edges: list[Edge], output_path: Path) -> None:
    ranked_nodes = sorted(nodes.values(), key=lambda item: (-item.frequency, item.name))
    names = [node.name for node in ranked_nodes]
    positions = spring_layout(names, edges, SVG_WIDTH, SVG_HEIGHT, iterations=260)
    max_frequency = max(node.frequency for node in ranked_nodes)
    max_weight = max(edge.weight for edge in edges) if edges else 1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "  <style>",
        "    .title { font: 700 34px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .subtitle { font: 500 18px 'Malgun Gothic', sans-serif; fill: #4b5563; }",
        "    .label { font: 600 11px 'Malgun Gothic', sans-serif; fill: #111827; }",
        "    .legend { font: 500 16px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
        '  <text class="title" x="60" y="70">\ubd88\uba74\uc99d \uacbd\ud608 \uacf5\ucd9c\ud604 \ub124\ud2b8\uc6cc\ud06c</text>',
        '  <text class="subtitle" x="60" y="105">\uc804\uccb4 163\uac1c \ub178\ub4dc \uae30\uc900 \uc2dc\uac01\ud654</text>',
    ]

    for edge in edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        opacity = 0.04 + 0.18 * (edge.weight / max_weight)
        stroke_width = 0.3 + 1.8 * (edge.weight / max_weight)
        lines.append(
            f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#94a3b8" stroke-opacity="{opacity:.2f}" stroke-width="{stroke_width:.2f}"/>'
        )

    label_nodes = ranked_nodes[:40]
    label_names = {node.name for node in label_nodes}

    for node in ranked_nodes:
        x, y = positions[node.name]
        radius = 3.0 + 13.0 * math.sqrt(node.frequency / max_frequency)
        fill = "#dc2626" if node.system == "ear" else "#2563eb"
        lines.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" fill-opacity="0.78" stroke="#ffffff" stroke-width="1.2"/>'
        )
        if node.name in label_names:
            lines.append(
                f'  <text class="label" x="{x:.2f}" y="{y + radius + 14:.2f}" text-anchor="middle">{xml_escape(node.name)}</text>'
            )

    lines.extend(
        [
            '  <circle cx="1280" cy="90" r="12" fill="#2563eb" fill-opacity="0.85"/>',
            '  <text class="legend" x="1305" y="96">\uccb4\uce68 \uacbd\ud608</text>',
            '  <circle cx="1280" cy="125" r="12" fill="#dc2626" fill-opacity="0.85"/>',
            '  <text class="legend" x="1305" y="131">\uc774\uce68 \uacbd\ud608</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def rank_top_items(values: dict[str, float], count: int = 5) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:count]


def build_unweighted_adjacency(
    nodes: dict[str, Node], edges: list[Edge]
) -> dict[str, set[str]]:
    adjacency = {name: set() for name in nodes}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    return adjacency


def build_weighted_adjacency(
    nodes: dict[str, Node], edges: list[Edge]
) -> dict[str, dict[str, float]]:
    adjacency = {name: {} for name in nodes}
    for edge in edges:
        adjacency[edge.source][edge.target] = float(edge.weight)
        adjacency[edge.target][edge.source] = float(edge.weight)
    return adjacency


def compute_degree_centrality(adjacency: dict[str, set[str]]) -> dict[str, float]:
    node_count = len(adjacency)
    if node_count <= 1:
        return {name: 0.0 for name in adjacency}
    return {
        name: len(neighbors) / (node_count - 1)
        for name, neighbors in adjacency.items()
    }


def compute_unweighted_closeness(adjacency: dict[str, set[str]]) -> dict[str, float]:
    node_count = len(adjacency)
    closeness: dict[str, float] = {}

    for start in adjacency:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        total_distance = sum(distances.values())
        reachable = len(distances)
        if total_distance > 0.0 and node_count > 1:
            score = (reachable - 1.0) / total_distance
            score *= (reachable - 1.0) / (node_count - 1.0)
        else:
            score = 0.0
        closeness[start] = score

    return closeness


def compute_unweighted_betweenness(adjacency: dict[str, set[str]]) -> dict[str, float]:
    nodes = sorted(adjacency)
    node_count = len(nodes)
    betweenness = dict.fromkeys(nodes, 0.0)

    for start in nodes:
        stack: list[str] = []
        predecessors = {node: [] for node in nodes}
        path_counts = dict.fromkeys(nodes, 0.0)
        path_counts[start] = 1.0
        distances = dict.fromkeys(nodes, -1)
        distances[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            stack.append(current)
            for neighbor in adjacency[current]:
                if distances[neighbor] < 0:
                    queue.append(neighbor)
                    distances[neighbor] = distances[current] + 1
                if distances[neighbor] == distances[current] + 1:
                    path_counts[neighbor] += path_counts[current]
                    predecessors[neighbor].append(current)

        dependency = dict.fromkeys(nodes, 0.0)
        while stack:
            current = stack.pop()
            for predecessor in predecessors[current]:
                dependency[predecessor] += (
                    path_counts[predecessor] / path_counts[current]
                ) * (1.0 + dependency[current])
            if current != start:
                betweenness[current] += dependency[current]

    if node_count <= 2:
        return betweenness

    scale = 1.0 / ((node_count - 1) * (node_count - 2) / 2)
    for node in betweenness:
        betweenness[node] *= 0.5 * scale
    return betweenness


def compute_strength(weighted_adjacency: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        name: sum(neighbors.values()) for name, neighbors in weighted_adjacency.items()
    }


def compute_weighted_closeness(
    weighted_adjacency: dict[str, dict[str, float]]
) -> dict[str, float]:
    node_count = len(weighted_adjacency)
    closeness: dict[str, float] = {}

    for start in weighted_adjacency:
        distances = {start: 0.0}
        heap: list[tuple[float, str]] = [(0.0, start)]
        while heap:
            distance, current = heappop(heap)
            if distance > distances[current]:
                continue
            for neighbor, weight in weighted_adjacency[current].items():
                next_distance = distance + (1.0 / weight)
                if neighbor not in distances or next_distance < distances[neighbor]:
                    distances[neighbor] = next_distance
                    heappush(heap, (next_distance, neighbor))

        total_distance = sum(distances.values())
        reachable = len(distances)
        if total_distance > 0.0 and node_count > 1:
            score = (reachable - 1.0) / total_distance
            score *= (reachable - 1.0) / (node_count - 1.0)
        else:
            score = 0.0
        closeness[start] = score

    return closeness


def compute_weighted_betweenness(
    weighted_adjacency: dict[str, dict[str, float]]
) -> dict[str, float]:
    nodes = sorted(weighted_adjacency)
    node_count = len(nodes)
    betweenness = dict.fromkeys(nodes, 0.0)

    for start in nodes:
        stack: list[str] = []
        predecessors = {node: [] for node in nodes}
        path_counts = dict.fromkeys(nodes, 0.0)
        path_counts[start] = 1.0
        distances = dict.fromkeys(nodes, float("inf"))
        distances[start] = 0.0
        heap: list[tuple[float, str]] = [(0.0, start)]

        while heap:
            current_distance, current = heappop(heap)
            if current_distance > distances[current]:
                continue
            stack.append(current)
            for neighbor, weight in weighted_adjacency[current].items():
                next_distance = distances[current] + (1.0 / weight)
                if next_distance < distances[neighbor] - 1e-12:
                    distances[neighbor] = next_distance
                    heappush(heap, (next_distance, neighbor))
                    path_counts[neighbor] = path_counts[current]
                    predecessors[neighbor] = [current]
                elif abs(next_distance - distances[neighbor]) <= 1e-12:
                    path_counts[neighbor] += path_counts[current]
                    predecessors[neighbor].append(current)

        dependency = dict.fromkeys(nodes, 0.0)
        while stack:
            current = stack.pop()
            for predecessor in predecessors[current]:
                dependency[predecessor] += (
                    path_counts[predecessor] / path_counts[current]
                ) * (1.0 + dependency[current])
            if current != start:
                betweenness[current] += dependency[current]

    if node_count <= 2:
        return betweenness

    scale = 1.0 / ((node_count - 1) * (node_count - 2) / 2)
    for node in betweenness:
        betweenness[node] *= 0.5 * scale
    return betweenness


def compute_centrality_rankings(
    nodes: dict[str, Node], edges: list[Edge]
) -> dict[str, list[tuple[str, float]]]:
    unweighted_adjacency = build_unweighted_adjacency(nodes, edges)
    weighted_adjacency = build_weighted_adjacency(nodes, edges)

    return {
        "degree_top5": rank_top_items(compute_degree_centrality(unweighted_adjacency)),
        "betweenness_top5": rank_top_items(
            compute_unweighted_betweenness(unweighted_adjacency)
        ),
        "closeness_top5": rank_top_items(
            compute_unweighted_closeness(unweighted_adjacency)
        ),
        "strength_top5": rank_top_items(compute_strength(weighted_adjacency)),
        "weighted_betweenness_top5": rank_top_items(
            compute_weighted_betweenness(weighted_adjacency)
        ),
        "weighted_closeness_top5": rank_top_items(
            compute_weighted_closeness(weighted_adjacency)
        ),
    }


def build_summary(nodes: dict[str, Node], edges: list[Edge]) -> str:
    top_node_rows = sorted(nodes.values(), key=lambda item: (-item.frequency, item.name))[:10]
    top_edge_rows = sorted(edges, key=lambda item: (-item.weight, item.source, item.target))[:10]
    ear_count = sum(1 for node in nodes.values() if node.system == "ear")
    body_count = sum(1 for node in nodes.values() if node.system == "body")
    adjacency: dict[str, set[str]] = {name: set() for name in nodes}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    edge_count = len(edges)
    node_count = len(nodes)
    possible_edges = node_count * (node_count - 1) / 2
    density = edge_count / possible_edges if possible_edges else 0.0

    components: list[list[str]] = []
    visited: set[str] = set()
    for node_name in nodes:
        if node_name in visited:
            continue
        queue = deque([node_name])
        visited.add(node_name)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    components.sort(key=len, reverse=True)
    largest_component = components[0] if components else []
    largest_component_set = set(largest_component)

    total_distance = 0
    distance_pairs = 0
    for start in largest_component:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in largest_component_set and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        for target, distance in distances.items():
            if target != start:
                total_distance += distance
                distance_pairs += 1

    average_shortest_path = total_distance / distance_pairs if distance_pairs else 0.0

    bottom_cluster_nodes = [
        "GV20",
        "GV24",
        "SP6",
        "PC6",
        "ST36",
        "BL15",
        "BL23",
        "BL62",
        "KI6",
        "LR3",
        "EX-HN1",
        "EX-HN22",
        "YINTANG",
        "ANMIAN",
    ]
    isolated_nodes = [component[0] for component in components if len(component) == 1]
    centrality_rankings = compute_centrality_rankings(nodes, edges)

    lines = [
        "\ubd88\uba74\uc99d \uacbd\ud608 \ub124\ud2b8\uc6cc\ud06c \uc694\uc57d",
        "",
        f"- \uc804\uccb4 \ub178\ub4dc \uc218: {len(nodes)}",
        f"- \uc804\uccb4 \uc5e3\uc9c0 \uc218: {len(edges)}",
        f"- \uccb4\uce68 \ub178\ub4dc \uc218: {body_count}",
        f"- \uc774\uce68 \ub178\ub4dc \uc218: {ear_count}",
        f"- \ucd5c\ub300 \uc5e3\uc9c0 \uac00\uc911\uce58: {max(edge.weight for edge in edges)}",
        f"- \ubc00\ub3c4(density): {density:.6f}",
        f"- \uc5f0\uacb0 \uc131\ubd84 \uc218: {len(components)}",
        f"- \ucd5c\ub300 \uc5f0\uacb0 \uc131\ubd84 \ub178\ub4dc \uc218: {len(largest_component)}",
        f"- \ud3c9\uade0 \ucd5c\ub2e8 \uacbd\ub85c \uae38\uc774: {average_shortest_path:.6f}",
        "- \ud3c9\uade0 \ucd5c\ub2e8 \uacbd\ub85c \uae38\uc774 \uae30\uc900: \uc804\uccb4 \uadf8\ub798\ud504\uac00 2\uac1c \uc131\ubd84\uc73c\ub85c \ub098\ub258\uc5b4 \uc788\uc5b4\uc11c, \uac00\uc7a5 \ud070 \uc5f0\uacb0 \uc131\ubd84(162\uac1c \ub178\ub4dc)\uae30\uc900\uc73c\ub85c \uacc4\uc0b0\ud568",
        "",
        "[\uc0c1\uc704 \ube48\ub3c4 \ub178\ub4dc 10\uac1c]",
    ]

    lines.extend(
        f"- {node.name}: {node.frequency}" for node in top_node_rows
    )
    lines.append("")
    lines.append("[\uc0c1\uc704 \uac00\uc911\uce58 \uc5e3\uc9c0 10\uac1c]")
    lines.extend(
        f"- {edge.source} - {edge.target}: {edge.weight}" for edge in top_edge_rows
    )
    lines.append("")
    lines.append("[\uc911\uc2ec\uc131 \ube44\uad50: \ube44\uac00\uc911\uce58 \uae30\uc900]")
    lines.append("- \uc5f0\uacb0 \uc911\uc2ec\uc131 \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.6f}"
        for name, value in centrality_rankings["degree_top5"]
    )
    lines.append("- \ub9e4\uac1c \uc911\uc2ec\uc131 \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.6f}"
        for name, value in centrality_rankings["betweenness_top5"]
    )
    lines.append("- \uadfc\uc811 \uc911\uc2ec\uc131 \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.6f}"
        for name, value in centrality_rankings["closeness_top5"]
    )
    lines.append("")
    lines.append("[\uc911\uc2ec\uc131 \ube44\uad50: \uac00\uc911\uce58 \uae30\uc900]")
    lines.append("- \uac00\uc911 \uc5f0\uacb0\uc131(strength) \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.0f}"
        for name, value in centrality_rankings["strength_top5"]
    )
    lines.append("- \uac00\uc911 \ub9e4\uac1c \uc911\uc2ec\uc131 \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.6f}"
        for name, value in centrality_rankings["weighted_betweenness_top5"]
    )
    lines.append("- \uac00\uc911 \uadfc\uc811 \uc911\uc2ec\uc131 \uc0c1\uc704 5\uac1c")
    lines.extend(
        f"  - {name}: {value:.6f}"
        for name, value in centrality_rankings["weighted_closeness_top5"]
    )
    lines.append("")
    lines.append("[\uc804\uccb4 SVG \ud558\ub2e8 \uad70\uc9d1 \ud574\uc11d]")
    lines.append("- `all_network_full.svg`\uc758 \uc544\ub798\ucabd \ubc30\uce58\ub294 \uc2e4\uc81c \uc704\uacc4\ub098 \ucd95 \uc758\ubbf8\uac00 \uc544\ub2c8\ub77c, \ud798 \uae30\ubc18 \ub124\ud2b8\uc6cc\ud06c \ubc30\uce58 \uacb0\uacfc\uc784")
    lines.append("- \uc544\ub798\ucabd\uc5d0 \ubab0\ub9b0 \ub178\ub4dc\ub4e4\uc740 \uc8fc\ub85c `GV20`, `GV24`, `SP6`, `PC6`, `ST36`, `BL15`, `BL23`, `BL62`, `KI6`, `LR3`, `EX-HN1`, `EX-HN22`, `YINTANG`, `ANMIAN` \ub4f1 \uccb4\uce68 \uad70\uc9d1\uc784")
    lines.append("- \uc774 \uad70\uc9d1\uc740 `GV20-SP6(19)`, `PC6-SP6(17)`, `GV20-PC6(16)`, `GV20-GV24(14)` \ucc98\ub7fc \uc11c\ub85c \uac15\ud558\uac8c \uc5f0\uacb0\ub41c \ub178\ub4dc\ub4e4\uc774 \uac00\uae4c\uc774 \ubaa8\uc778 \uac83\uc73c\ub85c \ubcfc \uc218 \uc788\uc74c")
    if isolated_nodes:
        lines.append(f"- \uace0\ub9bd \ub178\ub4dc: {', '.join(isolated_nodes)}")
    lines.append("")
    lines.append("[\uc2e4\uc2b5 3 \ub124\ud2b8\uc6cc\ud06c\uc640\uc758 \ucc28\uc774]")
    lines.append("- \uc2e4\uc2b5 3 \ub124\ud2b8\uc6cc\ud06c\ub294 \ub178\ub4dc 71\uac1c, \uc5e3\uc9c0 1,156\uac1c, \ubc00\ub3c4 0.4652, \uc5f0\uacb0 \uc131\ubd84 1\uac1c, \ud3c9\uade0 \ucd5c\ub2e8 \uacbd\ub85c \uae38\uc774 1.5533\uc774\uc5c8\uc74c")
    lines.append("- \ud604\uc7ac \uacfc\uc81c \ub124\ud2b8\uc6cc\ud06c\ub294 \ub178\ub4dc 163\uac1c, \uc5e3\uc9c0 5,285\uac1c\ub85c \uaddc\ubaa8\ub294 \ub354 \ud06c\uc9c0\ub9cc, \ubc00\ub3c4\ub294 0.400288\ub85c \uc2e4\uc2b5 3\ubcf4\ub2e4 \ub0ae\uc74c")
    lines.append("- \uc2e4\uc2b5 3\uc740 \uc804\uccb4\uac00 \ud558\ub098\uc758 \uc5f0\uacb0 \uc131\ubd84\uc73c\ub85c \ubd99\uc5b4 \uc788\ub294 \ub354 \uc751\uc9d1\uc801\uc778 \ub124\ud2b8\uc6cc\ud06c\uc600\uace0, \ud604\uc7ac \uacfc\uc81c\ub294 `EAR_ZHENJING` \uace0\ub9bd \ub178\ub4dc\ub97c \ud3ec\ud568\ud574 \uc5f0\uacb0 \uc131\ubd84\uc774 2\uac1c\uc784")
    lines.append("- \ud3c9\uade0 \ucd5c\ub2e8 \uacbd\ub85c \uae38\uc774\ub3c4 \ud604\uc7ac \uacfc\uc81c\uac00 1.695192\ub85c \uc2e4\uc2b5 3\uc758 1.5533\ubcf4\ub2e4 \uae38\uc5b4, \ub354 \ud070 \uaddc\ubaa8\uc5d0\uc11c \uc0c1\ub300\uc801\uc73c\ub85c \ub35c \uc555\ucd95\ub41c \uad6c\uc870\ub85c \ubcfc \uc218 \uc788\uc74c")
    lines.append("- \ud5c8\ube0c \uad6c\uc870\ub3c4 \ub2e4\ub978\ub370, \uc2e4\uc2b5 3\uc740 `BL23`, `BL40`, `BL25` \uc911\uc2ec\uc758 \uc694\ud1b5 \uacbd\ud608 \ucf54\uc5b4\uac00 \ub69c\ub837\ud588\uace0, \ud604\uc7ac \uacfc\uc81c\ub294 `HT7`, `GV20`, `SP6`, `PC6` \uc911\uc2ec\uc5d0 `EAR_SHENMEN` \uac19\uc740 \uc774\uce68 \ub178\ub4dc\uae4c\uc9c0 \uc0c1\uc704\uc5d0 \ud3ec\ud568\ub428")
    lines.append("- \uc989, \uc2e4\uc2b5 3\uc740 \ub2e8\uc77c \uc9c8\ud658 \uc911\uc2ec\uc758 \uc870\ubc00\ud55c \ucf54\uc5b4 \ub124\ud2b8\uc6cc\ud06c\uc5d0 \uac00\uae5d\uace0, \ud604\uc7ac \uacfc\uc81c\ub294 \uccb4\uce68\uacfc \uc774\uce68\uc774 \ud568\uaed8 \uc11e\uc778 \ub354 \ud070 \uaddc\ubaa8\uc758 \ub2e4\uc911 \uad70\uc9d1\ud615 \ub124\ud2b8\uc6cc\ud06c\ub85c \ud574\uc11d\ud560 \uc218 \uc788\uc74c")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    nodes = read_nodes(NODES_INPUT)
    edges = read_edges(EDGES_INPUT)

    write_graphml(nodes, edges, GRAPHML_OUTPUT)

    selected_nodes = top_nodes(nodes, TOP_NODE_COUNT)
    selected_names = {node.name for node in selected_nodes}
    selected_edges = filter_edges_for_nodes(edges, selected_names)
    write_svg(selected_nodes, selected_edges, SVG_OUTPUT)
    write_full_svg(nodes, edges, FULL_SVG_OUTPUT)

    summary_text = build_summary(nodes, edges)
    SUMMARY_OUTPUT.write_text(summary_text, encoding="utf-8")

    print(f"GraphML \uc800\uc7a5: {GRAPHML_OUTPUT}")
    print(f"SVG \uc800\uc7a5: {SVG_OUTPUT}")
    print(f"\uc804\uccb4 SVG \uc800\uc7a5: {FULL_SVG_OUTPUT}")
    print(f"\uc694\uc57d \uc800\uc7a5: {SUMMARY_OUTPUT}")
    print(f"\uc2dc\uac01\ud654 \ub178\ub4dc \uc218: {len(selected_nodes)}")
    print(f"\uc2dc\uac01\ud654 \uc5e3\uc9c0 \uc218: {len(selected_edges)}")


if __name__ == "__main__":
    main()
