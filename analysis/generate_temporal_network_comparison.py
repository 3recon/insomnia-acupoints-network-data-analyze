from __future__ import annotations

import ast
import csv
import sys
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.generate_network_graph import (
    Edge,
    Node,
    build_unweighted_adjacency,
    build_weighted_adjacency,
    compute_degree_centrality,
    compute_strength,
    compute_unweighted_betweenness,
    compute_unweighted_closeness,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_INPUT_PATHS = {
    "2012": DATA_DIR / "2012_clean.csv",
    "2016": DATA_DIR / "2016_clean.csv",
    "2020": DATA_DIR / "2020_clean.csv",
}
SUMMARY_OUTPUT = OUTPUT_DIR / "temporal_network_summary.csv"
TOP_NODES_OUTPUT = OUTPUT_DIR / "temporal_top_nodes.csv"
TOP_EDGES_OUTPUT = OUTPUT_DIR / "temporal_top_edges.csv"
REPORT_OUTPUT = OUTPUT_DIR / "temporal_network_report.txt"
TOP_NODES_BAR_CHART_OUTPUT = OUTPUT_DIR / "temporal_top10_nodes_bar_chart.svg"
TOP_NODES_PER_YEAR = 10
TOP_EDGES_PER_YEAR = 10
SVG_WIDTH = 1800
SVG_HEIGHT = 1200


@dataclass
class YearAnalysis:
    year: str
    paper_count: int
    nodes: dict[str, Node]
    edges: list[Edge]
    summary_row: dict[str, str]
    top_node_rows: list[dict[str, str]]
    top_edge_rows: list[dict[str, str]]


def read_paper_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list string, got: {value}")
    return [str(item) for item in parsed]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def infer_system(node: str) -> str:
    return "ear" if node.startswith("EAR_") else "body"


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def count_components(adjacency: dict[str, set[str]]) -> int:
    visited: set[str] = set()
    components = 0

    for start in adjacency:
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return components


def build_edges_and_nodes_from_rows(
    paper_rows: list[dict[str, str]],
) -> tuple[dict[str, Node], list[Edge]]:
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()

    for row in paper_rows:
        acupoints = unique_preserve_order(parse_list(row["standard_acupoints_list"]))

        for acupoint in acupoints:
            node_counter[acupoint] += 1

        for source, target in combinations(sorted(acupoints), 2):
            edge_counter[(source, target)] += 1

    nodes = {
        node_name: Node(
            name=node_name,
            system=infer_system(node_name),
            frequency=frequency,
        )
        for node_name, frequency in node_counter.items()
    }
    edges = [
        Edge(source=source, target=target, weight=weight)
        for (source, target), weight in edge_counter.items()
    ]
    edges.sort(key=lambda item: (-item.weight, item.source, item.target))
    return nodes, edges


def build_summary_row(
    year: str, paper_count: int, nodes: dict[str, Node], edges: list[Edge]
) -> dict[str, str]:
    node_count = len(nodes)
    edge_count = len(edges)
    possible_edges = node_count * (node_count - 1) / 2
    density = edge_count / possible_edges if possible_edges else 0.0
    avg_degree = (2 * edge_count / node_count) if node_count else 0.0
    avg_weight = sum(edge.weight for edge in edges) / edge_count if edge_count else 0.0

    body_count = sum(1 for node in nodes.values() if node.system == "body")
    ear_count = sum(1 for node in nodes.values() if node.system == "ear")
    body_ratio = body_count / node_count if node_count else 0.0
    ear_ratio = ear_count / node_count if node_count else 0.0

    adjacency = build_unweighted_adjacency(nodes, edges)

    return {
        "year": year,
        "paper_count": str(paper_count),
        "node_count": str(node_count),
        "edge_count": str(edge_count),
        "density": format_float(density),
        "avg_degree": format_float(avg_degree),
        "avg_weight": format_float(avg_weight),
        "component_count": str(count_components(adjacency)),
        "body_node_ratio": format_float(body_ratio),
        "ear_node_ratio": format_float(ear_ratio),
    }


def build_top_node_rows(
    year: str, nodes: dict[str, Node], edges: list[Edge]
) -> list[dict[str, str]]:
    unweighted_adjacency = build_unweighted_adjacency(nodes, edges)
    weighted_adjacency = build_weighted_adjacency(nodes, edges)
    degree_scores = compute_degree_centrality(unweighted_adjacency)
    strength_scores = compute_strength(weighted_adjacency)
    betweenness_scores = compute_unweighted_betweenness(unweighted_adjacency)
    closeness_scores = compute_unweighted_closeness(unweighted_adjacency)

    ranked_names = sorted(
        nodes,
        key=lambda name: (
            -strength_scores.get(name, 0.0),
            -degree_scores.get(name, 0.0),
            name,
        ),
    )[:TOP_NODES_PER_YEAR]

    rows: list[dict[str, str]] = []
    for rank, name in enumerate(ranked_names, start=1):
        node = nodes[name]
        rows.append(
            {
                "year": year,
                "rank": str(rank),
                "node": name,
                "system": node.system,
                "frequency": str(node.frequency),
                "degree_centrality": format_float(degree_scores.get(name, 0.0)),
                "strength": format_float(strength_scores.get(name, 0.0)),
                "betweenness_centrality": format_float(
                    betweenness_scores.get(name, 0.0)
                ),
                "closeness_centrality": format_float(closeness_scores.get(name, 0.0)),
            }
        )
    return rows


def build_top_edge_rows(year: str, edges: list[Edge]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rank, edge in enumerate(edges[:TOP_EDGES_PER_YEAR], start=1):
        rows.append(
            {
                "year": year,
                "rank": str(rank),
                "source": edge.source,
                "target": edge.target,
                "weight": str(edge.weight),
            }
        )
    return rows


def analyze_year(year: str, input_path: Path) -> YearAnalysis:
    paper_rows = read_paper_rows(input_path)
    nodes, edges = build_edges_and_nodes_from_rows(paper_rows)
    return YearAnalysis(
        year=year,
        paper_count=len(paper_rows),
        nodes=nodes,
        edges=edges,
        summary_row=build_summary_row(year, len(paper_rows), nodes, edges),
        top_node_rows=build_top_node_rows(year, nodes, edges),
        top_edge_rows=build_top_edge_rows(year, edges),
    )


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8-sig")
        return
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(year_analyses: list[YearAnalysis], output_path: Path) -> None:
    lines = [
        "연도별 경혈 공출현 네트워크 변화 분석",
        "",
    ]

    for analysis in year_analyses:
        strongest_node = analysis.top_node_rows[0]
        strongest_edge = analysis.top_edge_rows[0] if analysis.top_edge_rows else None
        summary = analysis.summary_row

        lines.extend(
            [
                f"[{analysis.year}]",
                f"- 논문 수: {summary['paper_count']}",
                f"- 노드 수: {summary['node_count']}",
                f"- 엣지 수: {summary['edge_count']}",
                f"- 밀도: {summary['density']}",
                f"- 평균 degree: {summary['avg_degree']}",
                f"- body 비율: {summary['body_node_ratio']}",
                f"- ear 비율: {summary['ear_node_ratio']}",
                (
                    f"- strength 기준 핵심 경혈: {strongest_node['node']} "
                    f"(strength={strongest_node['strength']})"
                ),
            ]
        )
        if strongest_edge is not None:
            lines.append(
                f"- 최강 공출현 조합: {strongest_edge['source']} - "
                f"{strongest_edge['target']} (weight={strongest_edge['weight']})"
            )
        lines.append("")

    if len(year_analyses) >= 2:
        lines.append("[변화 요약]")
        for previous, current in zip(year_analyses, year_analyses[1:]):
            lines.append(
                f"- {previous.year} -> {current.year}: "
                f"노드 {previous.summary_row['node_count']} -> {current.summary_row['node_count']}, "
                f"엣지 {previous.summary_row['edge_count']} -> {current.summary_row['edge_count']}, "
                f"ear 비율 {previous.summary_row['ear_node_ratio']} -> "
                f"{current.summary_row['ear_node_ratio']}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_top_nodes_bar_chart(
    year_analyses: list[YearAnalysis], output_path: Path
) -> None:
    panel_width = 500
    panel_height = 980
    panel_gap = 40
    panel_y = 110
    chart_left = 150
    chart_right = 40
    chart_top = 130
    chart_height = 860
    bar_gap = 12
    panel_starts = [50 + (panel_width + panel_gap) * index for index in range(len(year_analyses))]
    colors = {"body": "#2563eb", "ear": "#dc2626"}

    max_strength = max(
        float(row["strength"])
        for analysis in year_analyses
        for row in analysis.top_node_rows
    )
    max_strength = max(max_strength, 1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "  <style>",
        "    .title { font: 700 34px 'Malgun Gothic', sans-serif; fill: #111827; }",
        "    .subtitle { font: 500 18px 'Malgun Gothic', sans-serif; fill: #4b5563; }",
        "    .panel-title { font: 700 24px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .label { font: 600 15px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .value { font: 600 14px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "    .legend { font: 500 16px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
        '  <text class="title" x="50" y="60">연도별 Top 10 경혈 노드</text>',
        '  <text class="subtitle" x="50" y="95">strength 기준 상위 10개 노드 비교</text>',
        '  <circle cx="1500" cy="58" r="10" fill="#2563eb" fill-opacity="0.88"/>',
        '  <text class="legend" x="1520" y="64">body</text>',
        '  <circle cx="1600" cy="58" r="10" fill="#dc2626" fill-opacity="0.88"/>',
        '  <text class="legend" x="1620" y="64">ear</text>',
    ]

    for panel_x, analysis in zip(panel_starts, year_analyses):
        lines.extend(
            [
                f'  <rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" rx="18" fill="#ffffff" stroke="#dbe4ee"/>',
                f'  <text class="panel-title" x="{panel_x + 20}" y="{panel_y + 35}">{analysis.year}</text>',
            ]
        )

        rows = analysis.top_node_rows
        bar_area_width = panel_width - chart_left - chart_right
        slot_height = chart_height / max(len(rows), 1)
        bar_height = max(16.0, slot_height - bar_gap)

        for index, row in enumerate(rows):
            strength = float(row["strength"])
            y = chart_top + index * slot_height
            bar_width = (strength / max_strength) * bar_area_width
            color = colors.get(row["system"], "#64748b")

            lines.extend(
                [
                    f'  <text class="label" x="{panel_x + 20}" y="{y + bar_height * 0.7:.2f}">{index + 1}. {xml_escape(row["node"])}</text>',
                    f'  <rect x="{panel_x + chart_left}" y="{y:.2f}" width="{bar_area_width:.2f}" height="{bar_height:.2f}" fill="#e5e7eb" rx="6"/>',
                    f'  <rect x="{panel_x + chart_left}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" fill-opacity="0.88" rx="6"/>',
                    f'  <text class="value" x="{panel_x + chart_left + bar_width + 8:.2f}" y="{y + bar_height * 0.7:.2f}">{strength:.1f}</text>',
                ]
            )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8-sig")


def generate_temporal_comparison_artifacts(
    input_paths: dict[str, Path] = DEFAULT_INPUT_PATHS,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(exist_ok=True)

    ordered_years = sorted(input_paths)
    year_analyses = [analyze_year(year, input_paths[year]) for year in ordered_years]

    summary_rows = [analysis.summary_row for analysis in year_analyses]
    top_node_rows = [row for analysis in year_analyses for row in analysis.top_node_rows]
    top_edge_rows = [row for analysis in year_analyses for row in analysis.top_edge_rows]

    summary_path = output_dir / SUMMARY_OUTPUT.name
    top_nodes_path = output_dir / TOP_NODES_OUTPUT.name
    top_edges_path = output_dir / TOP_EDGES_OUTPUT.name
    report_path = output_dir / REPORT_OUTPUT.name
    bar_chart_path = output_dir / TOP_NODES_BAR_CHART_OUTPUT.name

    write_csv(summary_rows, summary_path)
    write_csv(top_node_rows, top_nodes_path)
    write_csv(top_edge_rows, top_edges_path)
    write_report(year_analyses, report_path)
    write_top_nodes_bar_chart(year_analyses, bar_chart_path)

    return summary_path, top_nodes_path, top_edges_path, report_path, bar_chart_path


def main() -> None:
    summary_path, top_nodes_path, top_edges_path, report_path, bar_chart_path = (
        generate_temporal_comparison_artifacts()
    )
    print(f"요약 저장: {summary_path}")
    print(f"상위 노드 저장: {top_nodes_path}")
    print(f"상위 엣지 저장: {top_edges_path}")
    print(f"보고서 저장: {report_path}")
    print(f"막대그래프 저장: {bar_chart_path}")


if __name__ == "__main__":
    main()
