from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import leidenalg

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.generate_network_graph import Node, read_edges, read_nodes, xml_escape
from analysis.generate_node2vec_embedding import (
    EmbeddingRow,
    RANDOM_SEED,
    RETURN_PARAMETER,
    INOUT_PARAMETER,
    WALK_LENGTH,
    NUM_WALKS,
    WINDOW_SIZE,
    NEGATIVE_SAMPLES,
    EPOCHS,
    LEARNING_RATE,
    build_weighted_adjacency,
    generate_walks,
    node_radius,
    read_embedding_csv,
    remove_isolates,
    scale_embeddings,
    train_skipgram_embeddings,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "clustering" / "results"
NODES_INPUT = DATA_DIR / "all_nodes.csv"
EDGES_INPUT = DATA_DIR / "all_edges.csv"
VIZ_INPUT = BASE_DIR / "results" / "node2vec_embedding_2d.csv"

HIGH_DIMENSIONS = 16
DEFAULT_K_NEIGHBORS = 6
DEFAULT_RESOLUTION = 0.7
K_GRID = [5, 8, 10, 12]
RESOLUTION_GRID = [0.8, 1.0, 1.2, 1.5]
SVG_WIDTH = 1600
SVG_HEIGHT = 1200

EMBEDDING_OUTPUT = OUTPUT_DIR / "node2vec_embedding_16d.csv"
CLUSTER_OUTPUT = OUTPUT_DIR / "node2vec_knn_leiden_clusters.csv"
SCATTER_OUTPUT = OUTPUT_DIR / "node2vec_knn_leiden_scatter.svg"
SUMMARY_OUTPUT = OUTPUT_DIR / "node2vec_knn_leiden_summary.txt"
COMPARE_OUTPUT = OUTPUT_DIR / "node2vec_knn_leiden_compare.txt"

CLUSTER_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#65a30d",
    "#4338ca",
    "#c2410c",
]


@dataclass
class ClusterRow:
    node: str
    system: str
    frequency: int
    cluster_id: int
    cluster_size: int
    viz_x: float
    viz_y: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left[index] * right[index] for index in range(len(left)))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return 0.0
    return numerator / (left_norm * right_norm)


def build_high_dim_embeddings(
    nodes_path: Path = NODES_INPUT,
    edges_path: Path = EDGES_INPUT,
    dimensions: int = HIGH_DIMENSIONS,
) -> tuple[dict[str, Node], dict[str, list[float]], list[str]]:
    nodes = read_nodes(nodes_path)
    edges = read_edges(edges_path)
    adjacency = build_weighted_adjacency(nodes, edges, use_edge_weights=True)
    filtered_nodes, filtered_adjacency, isolated_nodes = remove_isolates(nodes, adjacency)

    walks = generate_walks(
        adjacency=filtered_adjacency,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        rng=random.Random(RANDOM_SEED),
        return_parameter=RETURN_PARAMETER,
        inout_parameter=INOUT_PARAMETER,
    )
    embeddings = train_skipgram_embeddings(
        walks=walks,
        dimensions=dimensions,
        window_size=WINDOW_SIZE,
        negative_samples=NEGATIVE_SAMPLES,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
    )
    return filtered_nodes, embeddings, isolated_nodes


def write_high_dim_embedding_csv(
    nodes: dict[str, Node], embeddings: dict[str, list[float]], output_path: Path
) -> None:
    first_vector = next(iter(embeddings.values()))
    fieldnames = ["node", "system", "frequency"] + [
        f"dim_{index}" for index in range(1, len(first_vector) + 1)
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node_name in sorted(embeddings, key=lambda item: (-nodes[item].frequency, item)):
            row = {
                "node": node_name,
                "system": nodes[node_name].system,
                "frequency": nodes[node_name].frequency,
            }
            for index, value in enumerate(embeddings[node_name], start=1):
                row[f"dim_{index}"] = f"{value:.6f}"
            writer.writerow(row)


def build_knn_edges(
    embeddings: dict[str, list[float]], k_neighbors: int
) -> list[tuple[str, str, float]]:
    node_names = sorted(embeddings)
    merged_edges: dict[tuple[str, str], list[float]] = defaultdict(list)

    for source in node_names:
        similarities: list[tuple[str, float]] = []
        for target in node_names:
            if source == target:
                continue
            similarity = cosine_similarity(embeddings[source], embeddings[target])
            weight = max(0.0001, (similarity + 1.0) / 2.0)
            similarities.append((target, weight))

        similarities.sort(key=lambda item: (-item[1], item[0]))
        for target, weight in similarities[:k_neighbors]:
            edge_key = tuple(sorted((source, target)))
            merged_edges[edge_key].append(weight)

    return [
        (source, target, sum(weights) / len(weights))
        for (source, target), weights in sorted(merged_edges.items())
    ]


def build_knn_graph(embeddings: dict[str, list[float]], k_neighbors: int) -> ig.Graph:
    edges = build_knn_edges(embeddings, k_neighbors=k_neighbors)
    graph = ig.Graph()
    graph.add_vertices(sorted(embeddings))
    graph.add_edges([(source, target) for source, target, _ in edges])
    graph.es["weight"] = [weight for _, _, weight in edges]
    return graph


def run_leiden(
    graph: ig.Graph, resolution: float
) -> tuple[dict[str, int], float, dict[int, int]]:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=RANDOM_SEED,
    )
    memberships = {
        graph.vs[index]["name"]: cluster_id
        for index, cluster_id in enumerate(partition.membership)
    }
    cluster_sizes: dict[int, int] = defaultdict(int)
    for cluster_id in partition.membership:
        cluster_sizes[cluster_id] += 1
    return memberships, float(partition.modularity), dict(cluster_sizes)


def load_visualization_rows(path: Path = VIZ_INPUT) -> dict[str, EmbeddingRow]:
    return {row.node: row for row in read_embedding_csv(path)}


def build_cluster_rows(
    nodes: dict[str, Node],
    memberships: dict[str, int],
    cluster_sizes: dict[int, int],
    viz_rows: dict[str, EmbeddingRow],
) -> list[ClusterRow]:
    rows: list[ClusterRow] = []
    for node_name, cluster_id in memberships.items():
        if node_name not in viz_rows:
            continue
        viz_row = viz_rows[node_name]
        node = nodes[node_name]
        rows.append(
            ClusterRow(
                node=node_name,
                system=node.system,
                frequency=node.frequency,
                cluster_id=cluster_id,
                cluster_size=cluster_sizes[cluster_id],
                viz_x=viz_row.x,
                viz_y=viz_row.y,
            )
        )
    return sorted(rows, key=lambda row: (row.cluster_id, -row.frequency, row.node))


def write_cluster_csv(rows: list[ClusterRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node",
                "system",
                "frequency",
                "cluster_id",
                "cluster_size",
                "viz_x",
                "viz_y",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node": row.node,
                    "system": row.system,
                    "frequency": row.frequency,
                    "cluster_id": row.cluster_id,
                    "cluster_size": row.cluster_size,
                    "viz_x": f"{row.viz_x:.6f}",
                    "viz_y": f"{row.viz_y:.6f}",
                }
            )


def write_cluster_scatter_svg(
    rows: list[ClusterRow],
    k_neighbors: int,
    resolution: float,
    modularity: float,
    output_path: Path,
) -> None:
    base_rows = [
        EmbeddingRow(
            node=row.node,
            system=row.system,
            frequency=row.frequency,
            x=row.viz_x,
            y=row.viz_y,
        )
        for row in rows
    ]
    scaled_map = {row.node: row for row in scale_embeddings(base_rows)}
    max_frequency = max(row.frequency for row in rows)
    label_names = {row.node for row in sorted(rows, key=lambda item: (-item.frequency, item.node))[:25]}
    cluster_sizes: dict[int, int] = {row.cluster_id: row.cluster_size for row in rows}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "  <style>",
        "    .title { font: 700 34px 'Malgun Gothic', sans-serif; fill: #111827; }",
        "    .subtitle { font: 500 18px 'Malgun Gothic', sans-serif; fill: #4b5563; }",
        "    .label { font: 600 14px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .legend { font: 500 16px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "    .small { font: 500 13px 'Malgun Gothic', sans-serif; fill: #475569; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
        '  <text class="title" x="60" y="70">Node2Vec 16차원 k-NN Leiden 군집</text>',
        f'  <text class="subtitle" x="60" y="105">16차원 임베딩으로 k-NN 그래프를 만들고 Leiden을 적용한 뒤, 기존 2차원 좌표 위에 표시함 (k={k_neighbors}, resolution={resolution:.2f}, modularity={modularity:.4f})</text>',
    ]

    for row in sorted(rows, key=lambda item: item.cluster_size):
        scaled = scaled_map[row.node]
        fill = CLUSTER_COLORS[row.cluster_id % len(CLUSTER_COLORS)]
        lines.append(
            f'  <circle cx="{scaled.x:.2f}" cy="{scaled.y:.2f}" r="{node_radius(row.frequency, max_frequency):.2f}" '
            f'fill="{fill}" fill-opacity="0.80" stroke="#ffffff" stroke-width="1.8"/>'
        )
        if row.node in label_names:
            lines.append(
                f'  <text class="label" x="{scaled.x:.2f}" y="{scaled.y + 18:.2f}" text-anchor="middle">{xml_escape(row.node)}</text>'
            )

    legend_y = 150
    for cluster_id in sorted(cluster_sizes)[:10]:
        fill = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        lines.append(f'  <circle cx="1240" cy="{legend_y}" r="12" fill="{fill}" fill-opacity="0.85"/>')
        lines.append(
            f'  <text class="legend" x="1265" y="{legend_y + 6}">Cluster {cluster_id} (n={cluster_sizes[cluster_id]})</text>'
        )
        legend_y += 34

    lines.extend(
        [
            '  <text class="small" x="60" y="1140">좌표는 기존 2차원 Node2Vec, 색상은 16차원 k-NN Leiden 군집 결과</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_cluster_summary_text(
    rows: list[ClusterRow],
    isolated_nodes: list[str],
    k_neighbors: int,
    resolution: float,
    modularity: float,
) -> str:
    cluster_top_nodes: dict[int, list[tuple[str, int]]] = defaultdict(list)
    cluster_sizes: dict[int, int] = {}
    for row in rows:
        cluster_top_nodes[row.cluster_id].append((row.node, row.frequency))
        cluster_sizes[row.cluster_id] = row.cluster_size

    lines = [
        "Node2Vec 16차원 k-NN Leiden 요약",
        "",
        f"- 임베딩 차원: {HIGH_DIMENSIONS}",
        f"- k-NN 이웃 수: {k_neighbors}",
        f"- Leiden resolution: {resolution:.2f}",
        f"- 클러스터 수: {len(cluster_sizes)}",
        f"- modularity: {modularity:.6f}",
        f"- 제외된 고립 노드: {', '.join(isolated_nodes) if isolated_nodes else '없음'}",
        "",
    ]
    for cluster_id in sorted(cluster_sizes, key=lambda key: (-cluster_sizes[key], key)):
        top_nodes = ", ".join(
            f"{node}({frequency})"
            for node, frequency in sorted(
                cluster_top_nodes[cluster_id],
                key=lambda item: (-item[1], item[0]),
            )[:5]
        )
        lines.append(
            f"- {cluster_id}번 클러스터: 노드 {cluster_sizes[cluster_id]}개, 대표 경혈 {top_nodes}"
        )
    return "\n".join(lines) + "\n"


def build_resolution_comparison_text(results: list[dict[str, float | int]]) -> str:
    lines = [
        "Node2Vec 16차원 k-NN Leiden 비교",
        "",
        "k_neighbors\tresolution\tcluster_count\tmodularity\tlargest_cluster_size",
    ]
    for result in results:
        lines.append(
            f"{result['k_neighbors']}\t{result['resolution']:.2f}\t{result['cluster_count']}\t"
            f"{result['modularity']:.6f}\t{result['largest_cluster_size']}"
        )
    return "\n".join(lines) + "\n"


def evaluate_settings(
    embeddings: dict[str, list[float]],
    k_values: list[int] = K_GRID,
    resolutions: list[float] = RESOLUTION_GRID,
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for k_neighbors in k_values:
        graph = build_knn_graph(embeddings, k_neighbors=k_neighbors)
        for resolution in resolutions:
            _, modularity, cluster_sizes = run_leiden(graph, resolution=resolution)
            results.append(
                {
                    "k_neighbors": k_neighbors,
                    "resolution": resolution,
                    "cluster_count": len(cluster_sizes),
                    "modularity": modularity,
                    "largest_cluster_size": max(cluster_sizes.values()),
                }
            )
    return results


def generate_knn_leiden_artifacts(
    nodes_path: Path = NODES_INPUT,
    edges_path: Path = EDGES_INPUT,
    output_dir: Path = OUTPUT_DIR,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    resolution: float = DEFAULT_RESOLUTION,
) -> dict[str, Path]:
    nodes, embeddings, isolated_nodes = build_high_dim_embeddings(
        nodes_path=nodes_path,
        edges_path=edges_path,
        dimensions=HIGH_DIMENSIONS,
    )
    viz_rows = load_visualization_rows()
    graph = build_knn_graph(embeddings, k_neighbors=k_neighbors)
    memberships, modularity, cluster_sizes = run_leiden(graph, resolution=resolution)
    cluster_rows = build_cluster_rows(nodes, memberships, cluster_sizes, viz_rows)
    comparison_results = evaluate_settings(embeddings)

    output_dir.mkdir(exist_ok=True)
    embedding_csv = output_dir / EMBEDDING_OUTPUT.name
    cluster_csv = output_dir / CLUSTER_OUTPUT.name
    scatter_svg = output_dir / SCATTER_OUTPUT.name
    summary_txt = output_dir / SUMMARY_OUTPUT.name
    compare_txt = output_dir / COMPARE_OUTPUT.name

    write_high_dim_embedding_csv(nodes, embeddings, embedding_csv)
    write_cluster_csv(cluster_rows, cluster_csv)
    write_cluster_scatter_svg(
        cluster_rows,
        k_neighbors=k_neighbors,
        resolution=resolution,
        modularity=modularity,
        output_path=scatter_svg,
    )
    summary_txt.write_text(
        build_cluster_summary_text(
            cluster_rows,
            isolated_nodes=isolated_nodes,
            k_neighbors=k_neighbors,
            resolution=resolution,
            modularity=modularity,
        ),
        encoding="utf-8",
    )
    compare_txt.write_text(
        build_resolution_comparison_text(comparison_results),
        encoding="utf-8",
    )
    return {
        "embedding_csv": embedding_csv,
        "cluster_csv": cluster_csv,
        "scatter_svg": scatter_svg,
        "summary_txt": summary_txt,
        "compare_txt": compare_txt,
    }


def main() -> None:
    artifacts = generate_knn_leiden_artifacts()
    print(f"16차원 Node2Vec 저장: {artifacts['embedding_csv']}")
    print(f"k-NN Leiden 군집 저장: {artifacts['cluster_csv']}")
    print(f"k-NN Leiden 시각화 저장: {artifacts['scatter_svg']}")
    print(f"k-NN Leiden 요약 저장: {artifacts['summary_txt']}")
    print(f"k-NN Leiden 비교 저장: {artifacts['compare_txt']}")


if __name__ == "__main__":
    main()
