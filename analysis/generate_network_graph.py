from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
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


def build_summary(nodes: dict[str, Node], edges: list[Edge]) -> str:
    top_node_rows = sorted(nodes.values(), key=lambda item: (-item.frequency, item.name))[:10]
    top_edge_rows = sorted(edges, key=lambda item: (-item.weight, item.source, item.target))[:10]
    ear_count = sum(1 for node in nodes.values() if node.system == "ear")
    body_count = sum(1 for node in nodes.values() if node.system == "body")

    lines = [
        "\ubd88\uba74\uc99d \uacbd\ud608 \ub124\ud2b8\uc6cc\ud06c \uc694\uc57d",
        "",
        f"- \uc804\uccb4 \ub178\ub4dc \uc218: {len(nodes)}",
        f"- \uc804\uccb4 \uc5e3\uc9c0 \uc218: {len(edges)}",
        f"- \uccb4\uce68 \ub178\ub4dc \uc218: {body_count}",
        f"- \uc774\uce68 \ub178\ub4dc \uc218: {ear_count}",
        f"- \ucd5c\ub300 \uc5e3\uc9c0 \uac00\uc911\uce58: {max(edge.weight for edge in edges)}",
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
