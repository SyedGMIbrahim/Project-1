from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

# Empirical constants derived from Diseases_and_Symptoms_dataset.csv and SMOTE training distribution
DEFAULT_HALLMARK_FREQ_THRESHOLD = 0.40  # Symptom frequency >= 40% in disease cases qualifies as a hallmark
TRAINING_SPARSITY_MEAN = 5.87           # Empirical mean active-symptom count in SMOTE-augmented training rows
TRAINING_SPARSITY_STD = 1.67            # Empirical standard deviation in training rows
TOTAL_SCHEMA_FEATURES = 230             # Total symptom feature dimensions

# Decision gating thresholds
MIN_EVIDENCE_COVERAGE_DIAGNOSE = 0.35   # Minimum evidence coverage required to allow a definitive diagnosis
MIN_EVIDENCE_COVERAGE_CLARIFY = 0.15    # Coverage below 15% cannot be rescued by clarification -> ABSTAIN
MIN_HALLMARKS_PRESENT_DIAGNOSE = 1      # At least one disease hallmark must be present for definitive diagnosis
SPARSE_VECTOR_MAX_SYMPTOMS = 3          # Presentations with <= 3 symptoms are considered sparse
HIGH_SPARSITY_CONFIDENCE_PENALTY = 0.15 # Adaptive threshold markup for sparse inputs
MIN_CONFIDENCE_FLOOR = 0.65             # Absolute floor for model probability


class DiseaseHallmarkRegistry:
    """
    Empirical clinical knowledge base extracted directly from the training dataset.
    Calculates and indexes:
    1. Per-disease case counts N_d
    2. Empirical conditional symptom frequencies f_{d, j} = P(symptom_j = 1 | disease = d)
    3. Disease hallmark features (symptoms with f_{d, j} >= threshold) and their weights w_{d, j} = f_{d, j}
    """

    def __init__(self, csv_path: Path, hallmark_freq_threshold: float = DEFAULT_HALLMARK_FREQ_THRESHOLD):
        self.csv_path = csv_path
        self.hallmark_freq_threshold = hallmark_freq_threshold
        self.disease_counts: Dict[str, int] = {}
        self.symptom_frequencies: Dict[str, Dict[str, float]] = {}  # {disease: {symptom: freq}}
        self.hallmark_profiles: Dict[str, Dict[str, float]] = {}     # {disease: {hallmark_symptom: weight}}
        self.feature_columns: List[str] = []
        self._load_and_profile()

    def _load_and_profile(self) -> None:
        if not self.csv_path.exists():
            LOGGER.warning("Dataset CSV not found at %s. Hallmark profiles unavailable.", self.csv_path)
            return

        LOGGER.info("Profiling disease hallmarks from dataset: %s", self.csv_path)
        disease_counts = defaultdict(int)
        symptom_disease_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        with open(self.csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Locate disease target column
            target_idx = -1
            for i, col in enumerate(header):
                if col.lower().strip() in ("disease", "prognosis", "diseases"):
                    target_idx = i
                    break

            if target_idx == -1:
                LOGGER.error("Could not find disease column in CSV header: %s", header[:5])
                return

            self.feature_columns = [h.strip() for i, h in enumerate(header) if i != target_idx]

            for row in reader:
                if not row or len(row) <= target_idx:
                    continue
                disease = row[target_idx].strip().lower()
                if not disease:
                    continue
                disease_counts[disease] += 1
                for i, val in enumerate(row):
                    if i == target_idx or i >= len(header):
                        continue
                    v = val.strip()
                    if v in ("1", "1.0", "True", "true"):
                        symptom_disease_counts[disease][header[i].strip()] += 1

        self.disease_counts = dict(disease_counts)

        # Compute empirical frequencies and hallmark profiles
        for disease, n_cases in self.disease_counts.items():
            if n_cases == 0:
                continue
            freq_dict: Dict[str, float] = {}
            hallmarks: Dict[str, float] = {}
            for symptom, count in symptom_disease_counts[disease].items():
                freq = count / n_cases
                freq_dict[symptom] = freq
                if freq >= self.hallmark_freq_threshold:
                    hallmarks[symptom] = round(freq, 4)

            self.symptom_frequencies[disease] = freq_dict
            self.hallmark_profiles[disease] = hallmarks

        LOGGER.info(
            "Profiled %d diseases and %d symptom columns. Average hallmarks per disease: %.1f",
            len(self.disease_counts),
            len(self.feature_columns),
            sum(len(h) for h in self.hallmark_profiles.values()) / max(len(self.hallmark_profiles), 1),
        )

    def get_hallmarks(self, disease_name: str) -> Dict[str, float]:
        """Returns {symptom: weight} for the given disease."""
        return self.hallmark_profiles.get(disease_name.lower().strip(), {})

    def get_symptom_frequency(self, disease_name: str, symptom_name: str) -> float:
        """Returns conditional probability P(symptom | disease)."""
        freqs = self.symptom_frequencies.get(disease_name.lower().strip(), {})
        return freqs.get(symptom_name.strip(), 0.0)


def calculate_sparsity_metrics(
    active_count: int,
    total_features: int = TOTAL_SCHEMA_FEATURES,
    train_mean: float = TRAINING_SPARSITY_MEAN,
    train_std: float = TRAINING_SPARSITY_STD,
) -> Dict[str, Any]:
    """
    Computes live symptom representation sparsity and distributional domain mismatch.
    - Sparsity Score: S = 1 - (k / D)
    - Z-Score: z = (k - mu_train) / sigma_train
    """
    sparsity_score = round(1.0 - (active_count / max(total_features, 1)), 4)
    z_score = round((active_count - train_mean) / train_std, 2) if train_std > 0 else 0.0
    is_sparse = active_count <= SPARSE_VECTOR_MAX_SYMPTOMS

    return {
        "sparsity_score": sparsity_score,
        "active_symptom_count": active_count,
        "z_score": z_score,
        "is_sparse": is_sparse,
        "train_distribution": {"mean": train_mean, "std": train_std, "total_features": total_features},
    }


def calculate_evidence_coverage(
    candidate_disease: str,
    supported_symptoms: List[str],
    hallmark_registry: DiseaseHallmarkRegistry,
) -> Dict[str, Any]:
    """
    Computes the Diagnostic Evidence Coverage Score:
    EC = sum(w_i * present_i) / sum(w_i)
    Where w_i = empirical frequency of hallmark symptom i for the candidate disease.
    """
    hallmarks = hallmark_registry.get_hallmarks(candidate_disease)
    supported_set = set(s.strip().lower() for s in supported_symptoms)

    if not hallmarks:
        return {
            "evidence_coverage": 0.50 if supported_symptoms else 0.0,
            "total_hallmarks": 0,
            "present_hallmarks": [],
            "missing_hallmarks": [],
            "hallmark_coverage_ratio": 0.0,
        }

    total_weight = sum(hallmarks.values())
    matched_weight = 0.0
    present_hallmarks: List[Dict[str, Any]] = []
    missing_hallmarks: List[Dict[str, Any]] = []

    for hallmark, weight in sorted(hallmarks.items(), key=lambda item: item[1], reverse=True):
        if hallmark.strip().lower() in supported_set:
            matched_weight += weight
            present_hallmarks.append({"symptom": hallmark, "weight": weight})
        else:
            missing_hallmarks.append({"symptom": hallmark, "weight": weight})

    coverage = round(matched_weight / total_weight, 4) if total_weight > 0 else 0.0

    return {
        "evidence_coverage": coverage,
        "total_hallmarks": len(hallmarks),
        "present_hallmarks": present_hallmarks,
        "missing_hallmarks": missing_hallmarks,
        "hallmark_coverage_ratio": round(len(present_hallmarks) / len(hallmarks), 3) if hallmarks else 0.0,
    }


def select_highest_information_gain_question(
    top_candidates: List[Tuple[str, float]],
    supported_symptoms: List[str],
    uncertain_symptoms: List[str],
    hallmark_registry: DiseaseHallmarkRegistry,
    denied_symptoms: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Discriminative Information-Gain Clarification Engine:
    When the decision gate enters CLARIFY, selects the single most discriminative
    missing hallmark feature between competing hypotheses (d_1, d_2) to maximally
    reduce diagnostic uncertainty.

    Excludes already supported and denied (explicitly confirmed absent) symptoms.
    """
    if not top_candidates:
        return None

    supported_set = set(s.strip().lower() for s in supported_symptoms)
    denied_set = set(s.strip().lower() for s in (denied_symptoms or []))

    # 1. First priority: Resolve any UNCERTAIN near-tie or borderline symptoms not yet denied
    pending_uncertain = [u for u in uncertain_symptoms if u.strip().lower() not in denied_set]
    if pending_uncertain:
        first_uncertain = pending_uncertain[0]
        return {
            "target_symptom": first_uncertain,
            "question": f"Could you clarify if you are experiencing '{first_uncertain}'?",
            "discriminating_power": 1.0,
            "reason": "Resolving borderline semantic ambiguity directly observed in patient report",
        }

    # 2. Compare top-1 vs top-2 hypotheses (or top-1 vs background)
    top_disease = top_candidates[0][0]
    runner_up_disease = top_candidates[1][0] if len(top_candidates) > 1 else None

    top_hallmarks = hallmark_registry.get_hallmarks(top_disease)
    runner_hallmarks = hallmark_registry.get_hallmarks(runner_up_disease) if runner_up_disease else {}

    # Pool candidate discriminating features, excluding supported and denied symptoms
    candidate_features: Set[str] = set(top_hallmarks.keys()).union(runner_hallmarks.keys())
    candidate_features -= supported_set
    candidate_features -= denied_set

    if not candidate_features:
        return None

    # Calculate discriminative divergence: |f_{d1, j} - f_{d2, j}|
    scored_features: List[Tuple[str, float]] = []
    for feat in candidate_features:
        f1 = hallmark_registry.get_symptom_frequency(top_disease, feat)
        f2 = hallmark_registry.get_symptom_frequency(runner_up_disease, feat) if runner_up_disease else 0.0
        divergence = abs(f1 - f2)
        score = divergence * (f1 + 0.1)
        scored_features.append((feat, round(score, 3)))

    scored_features.sort(key=lambda x: x[1], reverse=True)
    best_symptom, best_score = scored_features[0]

    clean_name = best_symptom.replace("_", " ")
    question_text = f"Are you currently experiencing {clean_name}?"

    return {
        "target_symptom": best_symptom,
        "question": question_text,
        "discriminating_power": best_score,
        "competing_hypotheses": [top_disease, runner_up_disease] if runner_up_disease else [top_disease],
        "reason": f"Discriminates '{top_disease}' from alternative hypotheses",
    }


def evaluate_decision_gate(
    top_predictions: List[Tuple[str, float]],
    supported_symptoms: List[str],
    uncertain_symptoms: List[str],
    unsupported_symptoms: List[str],
    hallmark_registry: DiseaseHallmarkRegistry,
    denied_symptoms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Adaptive Multi-Criteria Decision Gate: T = f(S, A, C, N)
    Central principle: Model confidence is NOT evidence sufficiency.

    Evaluates:
    - S: Symptom sparsity & z-score
    - A: Semantic ambiguity (count of uncertain/unsupported symptoms)
    - C: Classifier softmax confidence
    - N: Number of independent supporting hallmarks

    Outputs:
    - decision: "DIAGNOSE" | "CLARIFY" | "ABSTAIN"
    - abstention_reason: Detailed clinical explanation if ABSTAIN
    - clarification_question: Targeted clarification query if CLARIFY
    - clarification_target: Target symptom key being queried
    - evidence_details: Full clinical evidence breakdown
    """
    active_count = len(supported_symptoms)
    sparsity_info = calculate_sparsity_metrics(active_count)
    denied_list = denied_symptoms or []

    if not top_predictions:
        return {
            "decision": "ABSTAIN",
            "predicted": "Non-specific symptoms",
            "confidence": 0.0,
            "evidence_coverage": 0.0,
            "abstention_reason": "No diagnostic hypotheses produced by classifier.",
            "clarification_question": None,
            "clarification_target": None,
            "sparsity_info": sparsity_info,
            "evidence_details": None,
        }

    top_disease, top_prob = top_predictions[0]
    coverage_info = calculate_evidence_coverage(top_disease, supported_symptoms, hallmark_registry)
    coverage = coverage_info["evidence_coverage"]
    present_hallmarks = coverage_info["present_hallmarks"]
    missing_hallmarks = coverage_info["missing_hallmarks"]

    # Dynamic adaptive threshold calculation
    adaptive_prob_floor = MIN_CONFIDENCE_FLOOR
    if sparsity_info["is_sparse"]:
        adaptive_prob_floor += HIGH_SPARSITY_CONFIDENCE_PENALTY  # e.g., 0.65 + 0.15 = 0.80

    decision = "DIAGNOSE"
    abstention_reason: Optional[str] = None
    clarification_question: Optional[str] = None
    clarification_target: Optional[str] = None

    # Filter missing hallmarks to prioritize non-denied ones in breakdown display
    denied_set = set(s.strip().lower() for s in denied_list)
    active_missing_hallmarks = [m for m in missing_hallmarks if m["symptom"].strip().lower() not in denied_set]

    # ── RULE 1: Severe Evidence Deficit or Complete Hallmarks Absence ──
    if len(present_hallmarks) < MIN_HALLMARKS_PRESENT_DIAGNOSE and coverage < MIN_EVIDENCE_COVERAGE_CLARIFY:
        decision = "ABSTAIN"
        abstention_reason = (
            f"Evidence Deficit: Model probability ({round(top_prob*100, 1)}%) for '{top_disease}' "
            f"disqualified by insufficient hallmark evidence coverage ({round(coverage*100, 1)}%). "
            f"Expected hallmark symptoms are completely absent from the clinical presentation."
        )

    # ── RULE 2: Sub-Floor Probability ──
    elif top_prob < MIN_CONFIDENCE_FLOOR:
        decision = "ABSTAIN"
        abstention_reason = (
            f"Low Confidence: Top candidate '{top_disease}' probability ({round(top_prob*100, 1)}%) "
            f"falls below the clinical safety threshold ({round(MIN_CONFIDENCE_FLOOR*100, 1)}%)."
        )

    # ── RULE 3: Moderate Coverage or Sparse Domain Mismatch -> CLARIFY ──
    elif coverage < MIN_EVIDENCE_COVERAGE_DIAGNOSE or (sparsity_info["is_sparse"] and top_prob < adaptive_prob_floor):
        q_info = select_highest_information_gain_question(
            top_predictions, supported_symptoms, uncertain_symptoms, hallmark_registry, denied_symptoms=denied_list
        )
        if q_info:
            decision = "CLARIFY"
            clarification_question = q_info["question"]
            clarification_target = q_info.get("target_symptom")
        else:
            decision = "ABSTAIN"
            denied_note = f" (denied absent: {', '.join(denied_list)})" if denied_list else ""
            abstention_reason = (
                f"Sparsity Mismatch: Presentation is sparse ({active_count} symptoms, z={sparsity_info['z_score']}) "
                f"with incomplete evidence coverage ({round(coverage*100, 1)}% < {round(MIN_EVIDENCE_COVERAGE_DIAGNOSE*100, 1)}%). "
                f"Clarification options exhausted{denied_note}."
            )

    # ── RULE 4: Borderline Semantic Ambiguity Exists ──
    elif uncertain_symptoms:
        q_info = select_highest_information_gain_question(
            top_predictions, supported_symptoms, uncertain_symptoms, hallmark_registry, denied_symptoms=denied_list
        )
        if q_info:
            decision = "CLARIFY"
            clarification_question = q_info["question"]
            clarification_target = q_info.get("target_symptom")

    evidence_details = {
        "candidate_disease": top_disease,
        "model_probability": round(top_prob, 4),
        "evidence_coverage": coverage,
        "adaptive_threshold": round(adaptive_prob_floor, 3),
        "present_hallmarks": present_hallmarks,
        "missing_hallmarks": active_missing_hallmarks[:5] if active_missing_hallmarks else missing_hallmarks[:5],
        "sparsity_metrics": sparsity_info,
        "epistemic_states": {
            "supported": supported_symptoms,
            "uncertain": uncertain_symptoms,
            "unsupported": unsupported_symptoms,
            "denied": denied_list,
        },
    }

    predicted_label = top_disease if decision == "DIAGNOSE" else f"{top_disease} (Evidence insufficient)"
    if decision == "ABSTAIN":
        predicted_label = f"Non-specific symptoms ({top_disease} suspected but unconfirmed)"

    return {
        "decision": decision,
        "predicted": predicted_label,
        "raw_predicted": top_disease,
        "confidence": top_prob,
        "evidence_coverage": coverage,
        "abstention_reason": abstention_reason,
        "clarification_question": clarification_question,
        "clarification_target": clarification_target,
        "sparsity_info": sparsity_info,
        "evidence_details": evidence_details,
    }

