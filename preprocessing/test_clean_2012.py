import unittest

from preprocessing.clean_2012 import normalize_acupoint


class NormalizeAcupointTest(unittest.TestCase):
    def test_keeps_standard_body_code(self):
        result = normalize_acupoint("HT7")

        self.assertEqual(result[0]["standard_acupoint"], "HT7")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "keep")

    def test_removes_side_information_from_body_code(self):
        result = normalize_acupoint("PC6(bilaterally)")

        self.assertEqual(result[0]["standard_acupoint"], "PC6")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_ear_acupoint(self):
        result = normalize_acupoint("(Ear) Shenmen")

        self.assertEqual(result[0]["standard_acupoint"], "EAR_SHENMEN")
        self.assertEqual(result[0]["system"], "ear")
        self.assertEqual(result[0]["action"], "map")

    def test_splits_combined_ear_acupoints(self):
        result = normalize_acupoint("occiput and subcortex")

        self.assertEqual(
            [item["standard_acupoint"] for item in result],
            ["EAR_OCCIPUT", "EAR_SUBCORTEX"],
        )
        self.assertTrue(all(item["action"] == "split" for item in result))

    def test_drops_unclear_value(self):
        result = normalize_acupoint("hands(unspecified)")

        self.assertEqual(result[0]["standard_acupoint"], "")
        self.assertEqual(result[0]["system"], "unknown")
        self.assertEqual(result[0]["action"], "drop")


if __name__ == "__main__":
    unittest.main()
