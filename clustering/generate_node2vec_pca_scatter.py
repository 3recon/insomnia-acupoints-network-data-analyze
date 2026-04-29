from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.generate_network_graph import xml_escape
from analysis.generate_node2vec_embedding import EmbeddingRow, node_radius, scale_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "clustering" / "results"
EMBEDDING_INPUT = OUTPUT_DIR / "node2vec_embedding_16d.csv"
CLUSTER_INPUT = OUTPUT_DIR / "node2vec_knn_leiden_clusters.csv"
PROJECTED_OUTPUT = OUTPUT_DIR / "node2vec_pca_2d_clusters.csv"
SVG_OUTPUT = OUTPUT_DIR / "node2vec_pca_2d_scatter.svg"

SVG_WIDTH = 1600
SVG_HEIGHT = 1200
RANDOM_SEED = 42
SCATTER_K_LABEL = "k=6"
SCATTER_RESOLUTION_LABEL = "resolution=0.70"
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
class HighDimRow:
    node: str
    system: str
    frequency: int
    vector: list[float]


@dataclass
class ClusterProjectionRow:
    node: str
    system: str
    frequency: int
    cluster_id: int
    pca_x: float
    pca_y: float


def read_high_dim_embedding_csv(path: Path) -> list[HighDimRow]:
    rows: list[HighDimRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        dimension_keys = [key for key in reader.fieldnames or [] if key.startswith("dim_")]
        for row in reader:
            rows.append(
                HighDimRow(
                    node=row["node"],
                    system=row["system"],
                    frequency=int(row["frequency"]),
                    vector=[float(row[key]) for key in dimension_keys],
                )
            )
    return rows


def read_cluster_ids(path: Path) -> dict[str, int]:
    cluster_ids: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster_ids[row["node"]] = int(row["cluster_id"])
    return cluster_ids


def center_vectors(vectors: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    dimension_count = len(vectors[0])
    means = [
        sum(vector[dimension] for vector in vectors) / len(vectors)
        for dimension in range(dimension_count)
    ]
    centered = [
        [vector[dimension] - means[dimension] for dimension in range(dimension_count)]
        for vector in vectors
    ]
    return centered, means


def covariance_matrix(centered_vectors: list[list[float]]) -> list[list[float]]:
    sample_count = len(centered_vectors)
    dimension_count = len(centered_vectors[0])
    scale = 1.0 / max(sample_count - 1, 1)
    matrix = [[0.0 for _ in range(dimension_count)] for _ in range(dimension_count)]
    for row in centered_vectors:
        for i in range(dimension_count):
            for j in range(dimension_count):
                matrix[i][j] += row[i] * row[j] * scale
    return matrix


def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row_index][column_index] * vector[column_index] for column_index in range(len(vector)))
        for row_index in range(len(matrix))
    ]


def dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(dot(vector, vector))
    if math.isclose(norm, 0.0):
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def principal_component(matrix: list[list[float]], seed: int) -> list[float]:
    rng = random.Random(seed)
    vector = normalize([rng.random() - 0.5 for _ in range(len(matrix))])
    for _ in range(100):
        next_vector = normalize(matrix_vector_multiply(matrix, vector))
        if max(abs(next_vector[index] - vector[index]) for index in range(len(vector))) < 1e-10:
            vector = next_vector
            break
        vector = next_vector
    return vector


def deflate_matrix(matrix: list[list[float]], component: list[float]) -> list[list[float]]:
    eigenvalue = dot(component, matrix_vector_multiply(matrix, component))
    size = len(matrix)
    return [
        [
            matrix[row][col] - eigenvalue * component[row] * component[col]
            for col in range(size)
        ]
        for row in range(size)
    ]


def compute_pca_projection(embeddings: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    node_names = sorted(embeddings)
    vectors = [embeddings[name] for name in node_names]
    centered_vectors, _ = center_vectors(vectors)
    matrix = covariance_matrix(centered_vectors)
    component1 = principal_component(matrix, seed=RANDOM_SEED)
    component2 = principal_component(deflate_matrix(matrix, component1), seed=RANDOM_SEED + 1)

    projection: dict[str, tuple[float, float]] = {}
    for node_name, centered in zip(node_names, centered_vectors):
        projection[node_name] = (dot(centered, component1), dot(centered, component2))
    return projection


def build_projection_rows(
    embedding_rows: list[HighDimRow], cluster_ids: dict[str, int]
) -> list[ClusterProjectionRow]:
    projection = compute_pca_projection({row.node: row.vector for row in embedding_rows})
    rows = [
        ClusterProjectionRow(
            node=row.node,
            system=row.system,
            frequency=row.frequency,
            cluster_id=cluster_ids[row.node],
            pca_x=projection[row.node][0],
            pca_y=projection[row.node][1],
        )
        for row in embedding_rows
        if row.node in cluster_ids
    ]
    return sorted(rows, key=lambda item: (item.cluster_id, -item.frequency, item.node))


def write_projected_csv(rows: list[ClusterProjectionRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node", "system", "frequency", "cluster_id", "pca_x", "pca_y"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node": row.node,
                    "system": row.system,
                    "frequency": row.frequency,
                    "cluster_id": row.cluster_id,
                    "pca_x": f"{row.pca_x:.6f}",
                    "pca_y": f"{row.pca_y:.6f}",
                }
            )


def write_pca_svg(rows: list[ClusterProjectionRow], output_path: Path) -> None:
    base_rows = [
        EmbeddingRow(
            node=row.node,
            system=row.system,
            frequency=row.frequency,
            x=row.pca_x,
            y=row.pca_y,
        )
        for row in rows
    ]
    scaled = {row.node: row for row in scale_embeddings(base_rows)}
    max_frequency = max(row.frequency for row in rows)
    cluster_sizes: dict[int, int] = {}
    for row in rows:
        cluster_sizes[row.cluster_id] = cluster_sizes.get(row.cluster_id, 0) + 1
    label_names = {
        row.node for row in sorted(rows, key=lambda item: (-item.frequency, item.node))[:25]
    }

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
        '  <text class="title" x="60" y="70">Node2Vec 16차원 PCA 2D 군집 시각화</text>',
        f'  <text class="subtitle" x="60" y="105">16차원 임베딩을 PCA 2D로 축소하고 기존 {SCATTER_K_LABEL}, {SCATTER_RESOLUTION_LABEL} Leiden 군집 색상을 표시함</text>',
    ]

    for row in sorted(rows, key=lambda item: cluster_sizes[item.cluster_id]):
        point = scaled[row.node]
        fill = CLUSTER_COLORS[row.cluster_id % len(CLUSTER_COLORS)]
        lines.append(
            f'  <circle cx="{point.x:.2f}" cy="{point.y:.2f}" r="{node_radius(row.frequency, max_frequency):.2f}" '
            f'fill="{fill}" fill-opacity="0.80" stroke="#ffffff" stroke-width="1.8"/>'
        )
        if row.node in label_names:
            lines.append(
                f'  <text class="label" x="{point.x:.2f}" y="{point.y + 18:.2f}" text-anchor="middle">{xml_escape(row.node)}</text>'
            )

    legend_y = 150
    for cluster_id in sorted(cluster_sizes):
        fill = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        lines.append(f'  <circle cx="1240" cy="{legend_y}" r="12" fill="{fill}" fill-opacity="0.85"/>')
        lines.append(
            f'  <text class="legend" x="1265" y="{legend_y + 6}">Cluster {cluster_id} (n={cluster_sizes[cluster_id]})</text>'
        )
        legend_y += 34

    lines.extend(
        [
            '  <text class="small" x="60" y="1140">좌표는 PCA 2D, 색상은 k-NN Leiden 군집 결과</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_pca_scatter_artifact(
    embedding_path: Path = EMBEDDING_INPUT,
    cluster_path: Path = CLUSTER_INPUT,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    embedding_rows = read_high_dim_embedding_csv(embedding_path)
    cluster_ids = read_cluster_ids(cluster_path)
    projection_rows = build_projection_rows(embedding_rows, cluster_ids)

    output_dir.mkdir(exist_ok=True)
    projected_csv_path = output_dir / PROJECTED_OUTPUT.name
    svg_path = output_dir / SVG_OUTPUT.name
    write_projected_csv(projection_rows, projected_csv_path)
    write_pca_svg(projection_rows, svg_path)
    return projected_csv_path, svg_path


def main() -> None:
    projected_csv_path, svg_path = generate_pca_scatter_artifact()
    print(f"PCA CSV 저장: {projected_csv_path}")
    print(f"PCA SVG 저장: {svg_path}")


if __name__ == "__main__":
    main()
