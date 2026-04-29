import csv
import tempfile
import unittest
from pathlib import Path

from analysis.generate_temporal_network_comparison import (
    DATA_DIR,
    generate_temporal_comparison_artifacts,
)


class GenerateTemporalNetworkComparisonTest(unittest.TestCase):
    def test_generates_temporal_comparison_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            summary_path, top_nodes_path, top_edges_path, report_path, bar_chart_path = (
                generate_temporal_comparison_artifacts(
                    input_paths={
                        "2012": DATA_DIR / "2012_clean.csv",
                        "2016": DATA_DIR / "2016_clean.csv",
                        "2020": DATA_DIR / "2020_clean.csv",
                    },
                    output_dir=output_dir,
                )
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(top_nodes_path.exists())
            self.assertTrue(top_edges_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(bar_chart_path.exists())

            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))

            self.assertEqual(["2012", "2016", "2020"], [row["year"] for row in summary_rows])
            self.assertEqual(
                {
                    "year",
                    "paper_count",
                    "node_count",
                    "edge_count",
                    "density",
                    "avg_degree",
                    "avg_weight",
                    "component_count",
                    "body_node_ratio",
                    "ear_node_ratio",
                },
                set(summary_rows[0].keys()),
            )

            with top_nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
                top_node_rows = list(csv.DictReader(handle))

            self.assertTrue(any(row["year"] == "2012" and row["rank"] == "1" for row in top_node_rows))
            self.assertEqual(
                {
                    "year",
                    "rank",
                    "node",
                    "system",
                    "frequency",
                    "degree_centrality",
                    "strength",
                    "betweenness_centrality",
                    "closeness_centrality",
                },
                set(top_node_rows[0].keys()),
            )

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("연도별 경혈 공출현 네트워크 변화 분석", report_text)
            self.assertIn("[2012]", report_text)
            self.assertIn("[2016]", report_text)
            self.assertIn("[2020]", report_text)

    def test_generates_top10_bar_chart_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            _, _, _, _, bar_chart_path = generate_temporal_comparison_artifacts(
                input_paths={
                    "2012": DATA_DIR / "2012_clean.csv",
                    "2016": DATA_DIR / "2016_clean.csv",
                    "2020": DATA_DIR / "2020_clean.csv",
                },
                output_dir=output_dir,
            )

            self.assertTrue(bar_chart_path.exists())

            svg_text = bar_chart_path.read_text(encoding="utf-8")
            self.assertIn("연도별 Top 10 경혈 노드", svg_text)
            self.assertIn("2012", svg_text)
            self.assertIn("2016", svg_text)
            self.assertIn("2020", svg_text)
            self.assertIn("GV20", svg_text)
            self.assertIn("<rect", svg_text)


if __name__ == "__main__":
    unittest.main()
