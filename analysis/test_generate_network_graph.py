import unittest

from analysis.generate_network_graph import (
    EDGES_INPUT,
    NODES_INPUT,
    compute_centrality_rankings,
    read_edges,
    read_nodes,
)


class CentralityRankingTest(unittest.TestCase):
    def test_computes_expected_top5_rankings_for_report(self):
        nodes = read_nodes(NODES_INPUT)
        edges = read_edges(EDGES_INPUT)

        rankings = compute_centrality_rankings(nodes, edges)

        self.assertEqual(
            [name for name, _ in rankings["degree_top5"]],
            ["GV20", "GV24", "BL23", "BL15", "BL62"],
        )
        self.assertEqual(
            [name for name, _ in rankings["betweenness_top5"]],
            ["GV20", "GV24", "BL62", "BL23", "CV12"],
        )
        self.assertEqual(
            [name for name, _ in rankings["closeness_top5"]],
            ["GV20", "GV24", "BL23", "BL15", "BL62"],
        )
        self.assertEqual(
            [name for name, _ in rankings["strength_top5"]],
            ["GV20", "HT7", "SP6", "BL15", "BL23"],
        )
        self.assertEqual(
            [name for name, _ in rankings["weighted_betweenness_top5"]],
            ["GV20", "HT7", "EAR_SHENMEN", "CV12", "SP6"],
        )
        self.assertEqual(
            [name for name, _ in rankings["weighted_closeness_top5"]],
            ["GV20", "HT7", "GV24", "SP6", "BL15"],
        )


if __name__ == "__main__":
    unittest.main()
