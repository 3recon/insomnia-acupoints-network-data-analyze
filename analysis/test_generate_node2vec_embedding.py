import csv
import tempfile
import unittest
from pathlib import Path

from analysis.generate_node2vec_embedding import (
    EDGES_INPUT,
    NODES_INPUT,
    build_weighted_adjacency,
    read_edges,
    read_nodes,
    generate_embedding_artifacts,
)


class GenerateNode2VecEmbeddingTest(unittest.TestCase):
    def test_generates_embedding_csv_and_svg_scatter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            csv_path, svg_path = generate_embedding_artifacts(
                nodes_path=NODES_INPUT,
                edges_path=EDGES_INPUT,
                output_dir=output_dir,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(svg_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertGreater(len(rows), 100)
            self.assertEqual(
                {"node", "system", "frequency", "x", "y"},
                set(rows[0].keys()),
            )

            isolated_rows = [row for row in rows if row["node"] == "EAR_ZHENJING"]
            self.assertEqual([], isolated_rows)

            for row in rows[:10]:
                float(row["x"])
                float(row["y"])

            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("Node2Vec 2D 임베딩", svg_text)
            self.assertIn("HT7", svg_text)

    def test_generates_unweighted_artifacts_with_distinct_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            csv_path, svg_path = generate_embedding_artifacts(
                nodes_path=NODES_INPUT,
                edges_path=EDGES_INPUT,
                output_dir=output_dir,
                use_edge_weights=False,
            )

            self.assertEqual("node2vec_embedding_2d_unweighted.csv", csv_path.name)
            self.assertEqual("node2vec_scatter_2d_unweighted.svg", svg_path.name)
            self.assertTrue(csv_path.exists())
            self.assertTrue(svg_path.exists())

    def test_unweighted_mode_flattens_all_edge_strengths(self):
        nodes = read_nodes(NODES_INPUT)
        edges = read_edges(EDGES_INPUT)

        weighted = build_weighted_adjacency(nodes, edges, use_edge_weights=True)
        unweighted = build_weighted_adjacency(nodes, edges, use_edge_weights=False)

        self.assertEqual(27.0, weighted["GV20"]["HT7"])
        self.assertEqual(1.0, unweighted["GV20"]["HT7"])
        self.assertEqual(1.0, unweighted["GV20"]["GV24"])


if __name__ == "__main__":
    unittest.main()
