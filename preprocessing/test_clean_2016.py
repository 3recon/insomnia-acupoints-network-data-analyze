import unittest

from preprocessing.clean_2016 import normalize_acupoint


class Normalize2016AcupointTest(unittest.TestCase):
    def test_maps_ear_point_with_prefix(self):
        result = normalize_acupoint("Ear points: heart")

        self.assertEqual(result[0]["standard_acupoint"], "EAR_HEART")
        self.assertEqual(result[0]["system"], "ear")
        self.assertEqual(result[0]["action"], "map")

    def test_splits_period_separated_body_points(self):
        result = normalize_acupoint("LR3. Yintang")

        self.assertEqual(
            [item["standard_acupoint"] for item in result],
            ["LR3", "YINTANG"],
        )
        self.assertTrue(all(item["action"] == "split" for item in result))

    def test_removes_treatment_prefix(self):
        result = normalize_acupoint("moxa:KI1")

        self.assertEqual(result[0]["standard_acupoint"], "KI1")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_named_body_extra_point(self):
        result = normalize_acupoint("Sishencong")

        self.assertEqual(result[0]["standard_acupoint"], "SISHENCONG")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")


if __name__ == "__main__":
    unittest.main()
