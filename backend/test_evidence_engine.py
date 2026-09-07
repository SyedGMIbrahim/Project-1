from __future__ import annotations

import sys
from pathlib import Path
import unittest

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evidence_engine import (
    DiseaseHallmarkRegistry,
    calculate_sparsity_metrics,
    calculate_evidence_coverage,
    select_highest_information_gain_question,
    evaluate_decision_gate,
)

DATASET_CSV_PATH = BASE_DIR / "data" / "train" / "Diseases_and_Symptoms_dataset.csv"


class TestEvidenceAdaptiveSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = DiseaseHallmarkRegistry(DATASET_CSV_PATH)

    def test_registry_initialization(self):
        """Verify 100 diseases and 230 features are indexed correctly."""
        self.assertEqual(len(self.registry.disease_counts), 100)
        self.assertEqual(len(self.registry.feature_columns), 230)
        # Ensure common conditions have hallmarks
        self.assertIn("conjunctivitis", self.registry.hallmark_profiles)
        self.assertIn("hyperemesis gravidarum", self.registry.hallmark_profiles)
        self.assertGreater(len(self.registry.get_hallmarks("conjunctivitis")), 0)

    def test_sparsity_metrics(self):
        """Verify sparsity score and training mismatch z-score calculations."""
        # 1 symptom: highly sparse
        s1 = calculate_sparsity_metrics(1)
        self.assertAlmostEqual(s1["sparsity_score"], 1.0 - (1 / 230), places=3)
        self.assertTrue(s1["is_sparse"])
        self.assertLess(s1["z_score"], -2.0)  # (1 - 5.87) / 1.67 = -2.92

        # 6 symptoms: close to training mean (5.87)
        s6 = calculate_sparsity_metrics(6)
        self.assertFalse(s6["is_sparse"])
        self.assertAlmostEqual(s6["z_score"], 0.08, places=1)

    def test_evidence_coverage_conjunctivitis(self):
        """Verify that a full hallmark presentation receives high evidence coverage."""
        eye_symptoms = ["itchiness of eye", "eye redness", "white discharge from eye", "lacrimation"]
        cov = calculate_evidence_coverage("conjunctivitis", eye_symptoms, self.registry)

        self.assertGreaterEqual(len(cov["present_hallmarks"]), 4)
        self.assertGreaterEqual(cov["evidence_coverage"], 0.35)

    def test_evidence_coverage_sparse_trap(self):
        """Verify that sparse input (headache + nausea) receives very low coverage for hyperemesis gravidarum."""
        sparse_symptoms = ["headache", "nausea"]
        cov = calculate_evidence_coverage("hyperemesis gravidarum", sparse_symptoms, self.registry)

        # Missing pregnancy and severe GI hallmarks
        self.assertLess(cov["evidence_coverage"], 0.25)
        missing_names = [m["symptom"] for m in cov["missing_hallmarks"]]
        self.assertTrue(any("pregnancy" in m for m in missing_names))

    def test_decision_gate_abstain_on_sparse_trap(self):
        """
        Verify the core patent claim: An 91% classifier probability on a sparse 2-symptom input
        is gated/clarified rather than allowed to produce a confident false diagnosis.
        """
        top_preds = [("hyperemesis gravidarum", 0.91), ("cholecystitis", 0.04)]
        gate = evaluate_decision_gate(
            top_predictions=top_preds,
            supported_symptoms=["headache", "nausea"],
            uncertain_symptoms=[],
            unsupported_symptoms=["strange visual aura", "photophobia"],
            hallmark_registry=self.registry,
        )

        # Must NOT be a flat DIAGNOSE
        self.assertIn(gate["decision"], ("CLARIFY", "ABSTAIN"))
        if gate["decision"] == "CLARIFY":
            self.assertIsNotNone(gate["clarification_question"])
            self.assertIn("?", gate["clarification_question"])
        else:
            self.assertIsNotNone(gate["abstention_reason"])

    def test_decision_gate_diagnose_on_supported_conjunctivitis(self):
        """Verify that high confidence + strong evidence coverage allows DIAGNOSE."""
        eye_symptoms = ["itchiness of eye", "eye redness", "white discharge from eye", "lacrimation"]
        top_preds = [("conjunctivitis", 0.99), ("conjunctivitis due to allergy", 0.01)]
        gate = evaluate_decision_gate(
            top_predictions=top_preds,
            supported_symptoms=eye_symptoms,
            uncertain_symptoms=[],
            unsupported_symptoms=[],
            hallmark_registry=self.registry,
        )

        self.assertEqual(gate["decision"], "DIAGNOSE")
        self.assertEqual(gate["predicted"], "conjunctivitis")
        self.assertGreaterEqual(gate["evidence_coverage"], 0.35)

    def test_information_gain_question_generation(self):
        """Verify optimal question generator picks the highest discriminative hallmark between competing diseases."""
        competing = [("acute sinusitis", 0.60), ("common cold", 0.35)]
        supported = ["headache", "fever"]
        q_info = select_highest_information_gain_question(
            top_candidates=competing,
            supported_symptoms=supported,
            uncertain_symptoms=[],
            hallmark_registry=self.registry,
        )

        self.assertIsNotNone(q_info)
        self.assertIn("target_symptom", q_info)
        self.assertIn("question", q_info)
        self.assertGreater(q_info["discriminating_power"], 0.0)


if __name__ == "__main__":
    unittest.main()

