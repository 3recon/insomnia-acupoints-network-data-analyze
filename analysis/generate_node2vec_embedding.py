from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.generate_network_graph import Edge, Node, read_edges, read_nodes, xml_escape


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
NODES_INPUT = DATA_DIR / "all_nodes.csv"
EDGES_INPUT = DATA_DIR / "all_edges.csv"
EMBEDDING_CSV_OUTPUT = OUTPUT_DIR / "node2vec_embedding_2d.csv"
SCATTER_SVG_OUTPUT = OUTPUT_DIR / "node2vec_scatter_2d.svg"
UNWEIGHTED_EMBEDDING_CSV_OUTPUT = OUTPUT_DIR / "node2vec_embedding_2d_unweighted.csv"
UNWEIGHTED_SCATTER_SVG_OUTPUT = OUTPUT_DIR / "node2vec_scatter_2d_unweighted.svg"
HT7_SIMILARITY_OUTPUT = OUTPUT_DIR / "ht7_top5_similarity_weighted.txt"

RANDOM_SEED = 42
EMBEDDING_DIMENSIONS = 2
WALK_LENGTH = 14
NUM_WALKS = 12
WINDOW_SIZE = 3
NEGATIVE_SAMPLES = 3
EPOCHS = 4
LEARNING_RATE = 0.035
RETURN_PARAMETER = 1.0
INOUT_PARAMETER = 0.8
SVG_WIDTH = 1600
SVG_HEIGHT = 1200


@dataclass
class EmbeddingRow:
    node: str
    system: str
    frequency: int
    x: float
    y: float


def build_weighted_adjacency(
    nodes: dict[str, Node], edges: list[Edge], use_edge_weights: bool = True
) -> dict[str, dict[str, float]]:
    adjacency = {name: {} for name in nodes}
    for edge in edges:
        weight = float(edge.weight) if use_edge_weights else 1.0
        adjacency[edge.source][edge.target] = weight
        adjacency[edge.target][edge.source] = weight
    return adjacency


def remove_isolates(
    nodes: dict[str, Node], adjacency: dict[str, dict[str, float]]
) -> tuple[dict[str, Node], dict[str, dict[str, float]], list[str]]:
    isolated = sorted(name for name, neighbors in adjacency.items() if not neighbors)
    filtered_nodes = {name: node for name, node in nodes.items() if name not in isolated}
    filtered_adjacency = {
        name: {neighbor: weight for neighbor, weight in neighbors.items() if neighbor in filtered_nodes}
        for name, neighbors in adjacency.items()
        if name in filtered_nodes
    }
    return filtered_nodes, filtered_adjacency, isolated


def transition_weight(
    previous: str | None,
    current: str,
    candidate: str,
    adjacency: dict[str, dict[str, float]],
    return_parameter: float,
    inout_parameter: float,
) -> float:
    base_weight = adjacency[current][candidate]
    if previous is None:
        return base_weight
    if candidate == previous:
        return base_weight / return_parameter
    if candidate in adjacency[previous]:
        return base_weight
    return base_weight / inout_parameter


def weighted_choice(rng: random.Random, weighted_items: list[tuple[str, float]]) -> str:
    total_weight = sum(weight for _, weight in weighted_items)
    threshold = rng.random() * total_weight
    cumulative = 0.0
    for item, weight in weighted_items:
        cumulative += weight
        if cumulative >= threshold:
            return item
    return weighted_items[-1][0]


def generate_biased_walk(
    start: str,
    adjacency: dict[str, dict[str, float]],
    walk_length: int,
    rng: random.Random,
    return_parameter: float,
    inout_parameter: float,
) -> list[str]:
    walk = [start]

    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = sorted(adjacency[current])
        if not neighbors:
            break

        previous = walk[-2] if len(walk) >= 2 else None
        weighted_neighbors = [
            (
                neighbor,
                transition_weight(
                    previous=previous,
                    current=current,
                    candidate=neighbor,
                    adjacency=adjacency,
                    return_parameter=return_parameter,
                    inout_parameter=inout_parameter,
                ),
            )
            for neighbor in neighbors
        ]
        walk.append(weighted_choice(rng, weighted_neighbors))

    return walk


def generate_walks(
    adjacency: dict[str, dict[str, float]],
    walk_length: int,
    num_walks: int,
    rng: random.Random,
    return_parameter: float,
    inout_parameter: float,
) -> list[list[str]]:
    walks: list[list[str]] = []
    node_names = sorted(adjacency)
    for _ in range(num_walks):
        shuffled = node_names[:]
        rng.shuffle(shuffled)
        for start in shuffled:
            walks.append(
                generate_biased_walk(
                    start=start,
                    adjacency=adjacency,
                    walk_length=walk_length,
                    rng=rng,
                    return_parameter=return_parameter,
                    inout_parameter=inout_parameter,
                )
            )
    return walks


def dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def update_pair(
    input_vector: list[float],
    output_vector: list[float],
    label: int,
    learning_rate: float,
) -> None:
    score = sigmoid(dot(input_vector, output_vector))
    gradient = (label - score) * learning_rate
    input_snapshot = input_vector[:]

    for index in range(len(input_vector)):
        input_vector[index] += gradient * output_vector[index]
        output_vector[index] += gradient * input_snapshot[index]


def build_negative_sampling_table(
    walks: list[list[str]], node_to_index: dict[str, int]
) -> list[int]:
    counts = [0.0] * len(node_to_index)
    for walk in walks:
        for node in walk:
            counts[node_to_index[node]] += 1.0

    weighted_counts = [count**0.75 for count in counts]
    total = sum(weighted_counts)
    if total == 0.0:
        return list(range(len(node_to_index)))

    table: list[int] = []
    scale = 10000
    for index, weight in enumerate(weighted_counts):
        repeats = max(1, int(round(scale * weight / total)))
        table.extend([index] * repeats)
    return table


def train_skipgram_embeddings(
    walks: list[list[str]],
    dimensions: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    vocabulary = sorted({node for walk in walks for node in walk})
    node_to_index = {node: index for index, node in enumerate(vocabulary)}
    index_to_node = {index: node for node, index in node_to_index.items()}

    input_vectors = [
        [(rng.random() - 0.5) / dimensions for _ in range(dimensions)]
        for _ in vocabulary
    ]
    output_vectors = [
        [0.0 for _ in range(dimensions)] for _ in vocabulary
    ]

    negative_table = build_negative_sampling_table(walks, node_to_index)
    if not negative_table:
        negative_table = list(range(len(vocabulary)))

    for _ in range(epochs):
        shuffled_walks = walks[:]
        rng.shuffle(shuffled_walks)

        for walk in shuffled_walks:
            for center_index, center_node in enumerate(walk):
                center_id = node_to_index[center_node]
                context_start = max(0, center_index - window_size)
                context_end = min(len(walk), center_index + window_size + 1)

                for context_index in range(context_start, context_end):
                    if context_index == center_index:
                        continue

                    context_id = node_to_index[walk[context_index]]
                    update_pair(
                        input_vector=input_vectors[center_id],
                        output_vector=output_vectors[context_id],
                        label=1,
                        learning_rate=learning_rate,
                    )

                    sampled = 0
                    while sampled < negative_samples:
                        negative_id = negative_table[rng.randrange(len(negative_table))]
                        if negative_id == context_id:
                            continue
                        update_pair(
                            input_vector=input_vectors[center_id],
                            output_vector=output_vectors[negative_id],
                            label=0,
                            learning_rate=learning_rate,
                        )
                        sampled += 1

    return {
        index_to_node[index]: vector
        for index, vector in enumerate(input_vectors)
    }


def scale_embeddings(rows: list[EmbeddingRow]) -> list[EmbeddingRow]:
    xs = [row.x for row in rows]
    ys = [row.y for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, lower: float, upper: float, start: float, end: float) -> float:
        if math.isclose(lower, upper):
            return (start + end) / 2.0
        ratio = (value - lower) / (upper - lower)
        return start + ratio * (end - start)

    scaled_rows: list[EmbeddingRow] = []
    for row in rows:
        scaled_rows.append(
            EmbeddingRow(
                node=row.node,
                system=row.system,
                frequency=row.frequency,
                x=scale(row.x, min_x, max_x, SVG_WIDTH * 0.08, SVG_WIDTH * 0.92),
                y=scale(row.y, min_y, max_y, SVG_HEIGHT * 0.88, SVG_HEIGHT * 0.14),
            )
        )
    return scaled_rows


def build_embedding_rows(
    nodes: dict[str, Node], embeddings: dict[str, list[float]]
) -> list[EmbeddingRow]:
    rows = [
        EmbeddingRow(
            node=name,
            system=node.system,
            frequency=node.frequency,
            x=embeddings[name][0],
            y=embeddings[name][1],
        )
        for name, node in nodes.items()
        if name in embeddings
    ]
    return sorted(rows, key=lambda row: (-row.frequency, row.node))


def write_embedding_csv(rows: list[EmbeddingRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node", "system", "frequency", "x", "y"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node": row.node,
                    "system": row.system,
                    "frequency": row.frequency,
                    "x": f"{row.x:.6f}",
                    "y": f"{row.y:.6f}",
                }
            )


def node_radius(frequency: int, max_frequency: int) -> float:
    if max_frequency <= 0:
        return 6.0
    return 4.0 + 12.0 * math.sqrt(frequency / max_frequency)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left[index] * right[index] for index in range(len(left)))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return 0.0
    return numerator / (left_norm * right_norm)


def read_embedding_csv(path: Path) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                EmbeddingRow(
                    node=row["node"],
                    system=row["system"],
                    frequency=int(row["frequency"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                )
            )
    return rows


def top_cosine_similarities(
    rows: list[EmbeddingRow], target_node: str, top_k: int
) -> list[tuple[str, float]]:
    vectors = {row.node: [row.x, row.y] for row in rows}
    if target_node not in vectors:
        raise ValueError(f"대상 노드가 임베딩에 없습니다: {target_node}")

    target_vector = vectors[target_node]
    scores: list[tuple[str, float]] = []
    for node, vector in vectors.items():
        if node == target_node:
            continue
        scores.append((node, cosine_similarity(target_vector, vector)))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:top_k]


def build_ht7_similarity_report_text(rows: list[EmbeddingRow]) -> str:
    top5 = top_cosine_similarities(rows, target_node="HT7", top_k=5)
    lines = [
        "HT7와 코사인 유사도가 높은 경혈 상위 5개",
        "",
        "- 기준 임베딩: 가중치 반영 Node2Vec 2차원 임베딩",
        "- 유사도 지표: 코사인 유사도",
        "",
    ]
    for index, (node, score) in enumerate(top5, start=1):
        lines.append(f"{index}. {node}: 코사인 유사도 {score:.6f}")
    return "\n".join(lines) + "\n"


def generate_ht7_similarity_report(
    embedding_csv_path: Path,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    rows = read_embedding_csv(embedding_csv_path)
    report_text = build_ht7_similarity_report_text(rows)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / HT7_SIMILARITY_OUTPUT.name
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def write_scatter_svg(
    rows: list[EmbeddingRow],
    output_path: Path,
    isolated_nodes: list[str],
    use_edge_weights: bool,
) -> None:
    scaled_rows = scale_embeddings(rows)
    max_frequency = max(row.frequency for row in scaled_rows)
    label_rows = scaled_rows[:20]
    label_names = {row.node for row in label_rows}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "  <style>",
        "    .title { font: 700 34px 'Malgun Gothic', sans-serif; fill: #111827; }",
        "    .subtitle { font: 500 18px 'Malgun Gothic', sans-serif; fill: #4b5563; }",
        "    .label { font: 600 14px 'Malgun Gothic', sans-serif; fill: #1f2937; }",
        "    .legend { font: 500 16px 'Malgun Gothic', sans-serif; fill: #374151; }",
        "    .axis { stroke: #cbd5e1; stroke-width: 1.2; }",
        "    .grid { stroke: #e5e7eb; stroke-width: 1; stroke-dasharray: 6 8; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
        '  <text class="title" x="60" y="70">불면증 경혈 네트워크 Node2Vec 2D 임베딩</text>',
        f'  <text class="subtitle" x="60" y="105">{xml_escape(build_subtitle(use_edge_weights))}</text>',
    ]

    grid_x = [280, 560, 840, 1120, 1400]
    grid_y = [220, 420, 620, 820, 1020]
    for x in grid_x:
        lines.append(f'  <line class="grid" x1="{x}" y1="140" x2="{x}" y2="1080"/>')
    for y in grid_y:
        lines.append(f'  <line class="grid" x1="120" y1="{y}" x2="1480" y2="{y}"/>')

    lines.extend(
        [
            '  <line class="axis" x1="120" y1="1080" x2="1480" y2="1080"/>',
            '  <line class="axis" x1="120" y1="1080" x2="120" y2="140"/>',
            '  <text class="legend" x="1420" y="1115" text-anchor="end">Dimension 1</text>',
            '  <text class="legend" x="78" y="165" transform="rotate(-90 78 165)">Dimension 2</text>',
        ]
    )

    for row in scaled_rows:
        fill = "#2563eb" if row.system == "body" else "#dc2626"
        lines.append(
            f'  <circle cx="{row.x:.2f}" cy="{row.y:.2f}" r="{node_radius(row.frequency, max_frequency):.2f}" '
            f'fill="{fill}" fill-opacity="0.78" stroke="#ffffff" stroke-width="1.8"/>'
        )
        if row.node in label_names:
            lines.append(
                f'  <text class="label" x="{row.x:.2f}" y="{row.y + 18:.2f}" text-anchor="middle">{xml_escape(row.node)}</text>'
            )

    isolated_text = ", ".join(isolated_nodes) if isolated_nodes else "없음"
    lines.extend(
        [
            '  <circle cx="1240" cy="92" r="12" fill="#2563eb" fill-opacity="0.82"/>',
            '  <text class="legend" x="1265" y="98">체침 경혈</text>',
            '  <circle cx="1240" cy="128" r="12" fill="#dc2626" fill-opacity="0.82"/>',
            '  <text class="legend" x="1265" y="134">이침 경혈</text>',
            f'  <text class="legend" x="60" y="1140">제외된 고립 노드: {xml_escape(isolated_text)}</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_embedding_artifacts(
    nodes_path: Path = NODES_INPUT,
    edges_path: Path = EDGES_INPUT,
    output_dir: Path = OUTPUT_DIR,
    use_edge_weights: bool = True,
) -> tuple[Path, Path]:
    nodes = read_nodes(nodes_path)
    edges = read_edges(edges_path)
    adjacency = build_weighted_adjacency(nodes, edges, use_edge_weights=use_edge_weights)
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
        dimensions=EMBEDDING_DIMENSIONS,
        window_size=WINDOW_SIZE,
        negative_samples=NEGATIVE_SAMPLES,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
    )
    rows = build_embedding_rows(filtered_nodes, embeddings)

    output_dir.mkdir(exist_ok=True)
    if use_edge_weights:
        csv_path = output_dir / EMBEDDING_CSV_OUTPUT.name
        svg_path = output_dir / SCATTER_SVG_OUTPUT.name
    else:
        csv_path = output_dir / UNWEIGHTED_EMBEDDING_CSV_OUTPUT.name
        svg_path = output_dir / UNWEIGHTED_SCATTER_SVG_OUTPUT.name
    write_embedding_csv(rows, csv_path)
    write_scatter_svg(rows, svg_path, isolated_nodes, use_edge_weights)
    return csv_path, svg_path


def build_subtitle(use_edge_weights: bool) -> str:
    if use_edge_weights:
        return "가중치 반영 편향 랜덤 워크 기반 노드 임베딩 산점도"
    return "비가중치 편향 랜덤 워크 기반 노드 임베딩 산점도"


def main() -> None:
    weighted_csv_path, weighted_svg_path = generate_embedding_artifacts(use_edge_weights=True)
    unweighted_csv_path, unweighted_svg_path = generate_embedding_artifacts(use_edge_weights=False)
    similarity_report_path = generate_ht7_similarity_report(weighted_csv_path)
    print(f"가중치 임베딩 CSV 저장: {weighted_csv_path}")
    print(f"가중치 산점도 SVG 저장: {weighted_svg_path}")
    print(f"비가중치 임베딩 CSV 저장: {unweighted_csv_path}")
    print(f"비가중치 산점도 SVG 저장: {unweighted_svg_path}")
    print(f"HT7 유사도 보고서 저장: {similarity_report_path}")


if __name__ == "__main__":
    main()
