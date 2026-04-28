from __future__ import annotations

import ast
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "all_papers_clean.csv"
EDGES_OUTPUT = DATA_DIR / "all_edges.csv"
NODES_OUTPUT = DATA_DIR / "all_nodes.csv"


def parse_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list string, got: {value}")
    return [str(item) for item in parsed]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def infer_system(node: str) -> str:
    return "ear" if node.startswith("EAR_") else "body"


def build_edges_and_nodes(papers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()

    for _, row in papers_df.iterrows():
        acupoints = unique_preserve_order(parse_list(row["standard_acupoints_list"]))

        for acupoint in acupoints:
            node_counter[acupoint] += 1

        for source, target in combinations(sorted(acupoints), 2):
            edge_counter[(source, target)] += 1

    edges_df = pd.DataFrame(
        [
            {"source": source, "target": target, "weight": weight}
            for (source, target), weight in edge_counter.items()
        ]
    ).sort_values(["weight", "source", "target"], ascending=[False, True, True])

    nodes_df = pd.DataFrame(
        [
            {
                "node": node,
                "system": infer_system(node),
                "frequency": frequency,
            }
            for node, frequency in node_counter.items()
        ]
    ).sort_values(["frequency", "node"], ascending=[False, True])

    return edges_df.reset_index(drop=True), nodes_df.reset_index(drop=True)


def main() -> None:
    papers_df = pd.read_csv(INPUT_FILE)
    edges_df, nodes_df = build_edges_and_nodes(papers_df)

    edges_df.to_csv(EDGES_OUTPUT, index=False, encoding="utf-8-sig")
    nodes_df.to_csv(NODES_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved {len(edges_df)} edges to {EDGES_OUTPUT}")
    print(f"Saved {len(nodes_df)} nodes to {NODES_OUTPUT}")


if __name__ == "__main__":
    main()
