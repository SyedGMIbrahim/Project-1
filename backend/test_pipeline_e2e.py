from __future__ import annotations

import sys
from pathlib import Path

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

def run_e2e_verification():
    print("=" * 75)
    print("EVIDENCE-ADAPTIVE CLINICAL DECISION SUPPORT — BENCHMARK EVALUATION")
    print("=" * 75)

    registry = DiseaseHallmarkRegistry(DATASET_CSV_PATH)
    print(f"[OK] Indexed {len(registry.disease_counts)} diseases and {len(registry.feature_columns)} features.\n")

    test_cases = [
        {
            "name": "Case 1: Full-Hallmark Conjunctivitis Presentation",
            "symptoms": ["itchiness of eye", "eye redness", "white discharge from eye", "lacrimation"],
            "model_preds": [("conjunctivitis", 0.99), ("conjunctivitis due to allergy", 0.01)],
            "expected_decision": "DIAGNOSE",
            "expected_coverage_min": 0.35,
        },
        {
            "name": "Case 2: Sparse-Input Overconfidence Trap (Migraine -> Hyperemesis Gravidarum)",
            "symptoms": ["headache", "nausea"],
            "model_preds": [("hyperemesis gravidarum", 0.91), ("cholecystitis", 0.05)],
            "expected_decision": ("CLARIFY", "ABSTAIN"),
            "expected_coverage_max": 0.25,
        },
        {
            "name": "Case 3: Sparse Diabetic Presentation against BPH",
            "symptoms": ["frequent urination", "paresthesia"],
            "model_preds": [("benign prostatic hyperplasia (bph)", 0.88), ("diabetes", 0.05)],
            "expected_decision": ("CLARIFY", "ABSTAIN"),
            "expected_coverage_max": 0.25,
        },
        {
            "name": "Case 4: Gout with Sparse Toe Complaints",
            "symptoms": ["foot or toe pain", "foot or toe swelling"],
            "model_preds": [("gout", 0.53), ("arthritis of the hip", 0.20)],
            "expected_decision": "ABSTAIN",
            "expected_coverage_min": 0.0,
        },
    ]

    all_passed = True

    for tc in test_cases:
        print(f"--- {tc['name']} ---")
        top_disease, top_prob = tc["model_preds"][0]
        cov_info = calculate_evidence_coverage(top_disease, tc["symptoms"], registry)
        sparsity_info = calculate_sparsity_metrics(len(tc["symptoms"]))

        gate = evaluate_decision_gate(
            top_predictions=tc["model_preds"],
            supported_symptoms=tc["symptoms"],
            uncertain_symptoms=[],
            unsupported_symptoms=[],
            hallmark_registry=registry,
        )

        print(f"  Symptoms in Vector: {tc['symptoms']}")
        print(f"  Model Softmax Probability: {top_prob*100:.1f}% for '{top_disease}'")
        print(f"  Diagnostic Evidence Coverage: {cov_info['evidence_coverage']*100:.1f}%")
        print(f"  Sparsity Metric: S = {sparsity_info['sparsity_score']:.3f}, z = {sparsity_info['z_score']}")
        print(f"  Decision Gate Result: [{gate['decision']}] -> {gate['predicted']}")

        if gate["clarification_question"]:
            print(f"  Targeted Question: {gate['clarification_question']}")
        if gate["abstention_reason"]:
            print(f"  Abstention Rationale: {gate['abstention_reason']}")

        # Verify decision
        expected = tc["expected_decision"]
        if isinstance(expected, tuple):
            passed = gate["decision"] in expected
        else:
            passed = gate["decision"] == expected

        if not passed:
            print(f"  FAILED: Expected decision {expected}, got {gate['decision']}")
            all_passed = False
        else:
            print(f"  [PASS] Verified decision matches clinical expectation.")

        print()

    print("=" * 75)
    if all_passed:
        print("ALL BENCHMARK VERIFICATION TESTS PASSED PERFECTLY (100%)")
    else:
        print("SOME TESTS FAILED")
    print("=" * 75)

if __name__ == "__main__":
    run_e2e_verification()

