import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clustering.generate_node2vec_knn_leiden import (
    EDGES_INPUT,
    NODES_INPUT,
    build_knn_edges,
    build_resolution_comparison_text,
    generate_knn_leiden_artifacts,
)


class GenerateNode2VecKnnLeidenTest(unittest.TestCase):
    def test_build_knn_edges_returns_positive_similarity_edges(self):
        embeddings = {
            "A": [1.0, 0.0, 0.0],
            "B": [0.9, 0.1, 0.0],
            "C": [0.0, 1.0, 0.0],
            "D": [0.0, 0.9, 0.1],
        }

        edges = build_knn_edges(embeddings, k_neighbors=1)

        self.assertTrue(edges)
        self.assertTrue(all(edge[2] > 0.0 for edge in edges))
        node_pairs = {(min(source, target), max(source, target)) for source, target, _ in edges}
        self.assertIn(("A", "B"), node_pairs)
        self.assertIn(("C", "D"), node_pairs)

    def test_build_resolution_comparison_text_formats_table(self):
        text = build_resolution_comparison_text(
            [
                {
                    "k_neighbors": 8,
                    "resolution": 1.0,
                    "cluster_count": 4,
                    "modularity": 0.21,
                    "largest_cluster_size": 60,
                }
            ]
        )

        self.assertIn("k_neighbors", text)
        self.assertIn("1.00", text)
        self.assertIn("0.210000", text)

    def test_generates_embedding_cluster_and_comparison_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            artifacts = generate_knn_leiden_artifacts(
                nodes_path=NODES_INPUT,
                edges_path=EDGES_INPUT,
                output_dir=output_dir,
            )

            for path in artifacts.values():
                self.assertTrue(path.exists(), msg=str(path))

            with artifacts["embedding_csv"].open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 100)
            self.assertIn("dim_16", rows[0])

            with artifacts["cluster_csv"].open("r", encoding="utf-8-sig", newline="") as handle:
                cluster_rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len({row["cluster_id"] for row in cluster_rows}), 2)
            self.assertIn("viz_x", cluster_rows[0])
            self.assertIn("viz_y", cluster_rows[0])

            summary_text = artifacts["summary_txt"].read_text(encoding="utf-8")
            compare_text = artifacts["compare_txt"].read_text(encoding="utf-8")
            self.assertIn("k-NN", summary_text)
            self.assertIn("Leiden", summary_text)
            self.assertIn("resolution", compare_text)

    def test_script_runs_as_standalone(self):
        result = subprocess.run(
            [sys.executable, "clustering/generate_node2vec_knn_leiden.py"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("16차원 Node2Vec 저장", result.stdout)


if __name__ == "__main__":
    unittest.main()
