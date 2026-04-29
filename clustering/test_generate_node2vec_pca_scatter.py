import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clustering.generate_node2vec_pca_scatter import (
    CLUSTER_INPUT,
    EMBEDDING_INPUT,
    compute_pca_projection,
    generate_pca_scatter_artifact,
    read_high_dim_embedding_csv,
)


class GenerateNode2VecPcaScatterTest(unittest.TestCase):
    def test_compute_pca_projection_returns_two_coordinates(self):
        vectors = {
            "A": [1.0, 0.0, 0.0],
            "B": [0.8, 0.2, 0.0],
            "C": [0.0, 1.0, 0.0],
            "D": [0.0, 0.8, 0.2],
        }

        projection = compute_pca_projection(vectors)

        self.assertEqual(set(vectors), set(projection))
        for coords in projection.values():
            self.assertEqual(2, len(coords))

    def test_read_high_dim_embedding_csv_includes_dimensions(self):
        rows = read_high_dim_embedding_csv(EMBEDDING_INPUT)

        self.assertGreater(len(rows), 100)
        self.assertGreaterEqual(len(rows[0].vector), 16)

    def test_generates_svg_and_projected_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            projected_csv_path, svg_path = generate_pca_scatter_artifact(
                embedding_path=EMBEDDING_INPUT,
                cluster_path=CLUSTER_INPUT,
                output_dir=output_dir,
            )

            self.assertTrue(projected_csv_path.exists())
            self.assertTrue(svg_path.exists())

            with projected_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertGreater(len(rows), 100)
            self.assertEqual(
                {"node", "system", "frequency", "cluster_id", "pca_x", "pca_y"},
                set(rows[0].keys()),
            )

            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("PCA", svg_text)
            self.assertIn("k=6", svg_text)
            self.assertIn("resolution=0.70", svg_text)

    def test_script_runs_as_standalone(self):
        result = subprocess.run(
            [sys.executable, "clustering/generate_node2vec_pca_scatter.py"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("PCA SVG 저장", result.stdout)


if __name__ == "__main__":
    unittest.main()
