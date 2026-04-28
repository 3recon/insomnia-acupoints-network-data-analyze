import unittest

from preprocessing.clean_2020 import normalize_acupoint


class Normalize2020AcupointTest(unittest.TestCase):
    def test_maps_ear_shenmen_with_side_information(self):
        result = normalize_acupoint("Ear Shenmen(bilateral)")

        self.assertEqual(result[0]["standard_acupoint"], "EAR_SHENMEN")
        self.assertEqual(result[0]["system"], "ear")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_ear_code_suffix(self):
        result = normalize_acupoint("Heart(CO15)")

        self.assertEqual(result[0]["standard_acupoint"], "EAR_HEART")
        self.assertEqual(result[0]["system"], "ear")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_sympathetic_autonomic(self):
        result = normalize_acupoint("Sympathetic autonomic(AH6a)")

        self.assertEqual(result[0]["standard_acupoint"], "EAR_SYMPATHETIC")
        self.assertEqual(result[0]["system"], "ear")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_du_to_gv(self):
        result = normalize_acupoint("DU14")

        self.assertEqual(result[0]["standard_acupoint"], "GV14")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")

    def test_maps_rn_to_cv(self):
        result = normalize_acupoint("RN12")

        self.assertEqual(result[0]["standard_acupoint"], "CV12")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")

    def test_keeps_corrected_ex_hn1_typo_note(self):
        result = normalize_acupoint("EX-HN1(EX-HN2은 논문 오타인듯)")

        self.assertEqual(result[0]["standard_acupoint"], "EX-HN1")
        self.assertEqual(result[0]["system"], "body")
        self.assertEqual(result[0]["action"], "map")


if __name__ == "__main__":
    unittest.main()
