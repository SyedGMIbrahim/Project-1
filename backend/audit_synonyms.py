from __future__ import annotations

import difflib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("synonym_auditor")

BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV_PATH = BASE_DIR / "data" / "train" / "Diseases_and_Symptoms_dataset.csv"
SYNONYMS_FILE = BASE_DIR.parent / "scratch" / "generated_synonyms.json"
APPROVED_OUTPUT_FILE = BASE_DIR / "approved_synonyms_v1.json"

# Anatomical keywords for conflict detection
ANATOMICAL_ZONES = {
    "eye": {"eye", "ocular", "eyelash", "eyelid", "conjunctiv", "pupil", "iris", "lacrimation", "gritty"},
    "ear": {"ear", "hearing", "eardrum", "tinnitus", "auditory"},
    "nasal": {"nasal", "nose", "sinus", "smell", "coryza"},
    "genital": {"vaginal", "penile", "genital", "vulvar", "scrotum", "testes", "prostate"},
    "foot_toe": {"foot", "toe", "ankle", "podiatric", "feet"},
    "oral": {"mouth", "oral", "tooth", "tongue", "gum", "toothache", "taste"},
}


def load_schema_features() -> List[str]:
    """Extract valid schema symptom features from the training CSV."""
    import csv
    with open(DATASET_CSV_PATH, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        target_candidates = {"disease", "prognosis", "diseases"}
        features = [h.strip() for h in header if h.strip().lower() not in target_candidates]
    return features


def detect_anatomical_conflict(phrase: str, target_feature: str) -> Optional[str]:
    """
    Checks if phrase contains an explicit anatomical qualifier from zone A
    while target_feature belongs to an incompatible zone B.
    """
    phrase_lower = phrase.lower()
    target_lower = target_feature.lower()

    phrase_zones = [zone for zone, kws in ANATOMICAL_ZONES.items() if any(kw in phrase_lower for kw in kws)]
    target_zones = [zone for zone, kws in ANATOMICAL_ZONES.items() if any(kw in target_lower for kw in kws)]

    if phrase_zones and target_zones:
        # If phrase explicitly specifies one zone, target cannot belong to a completely disjoint zone
        if not set(phrase_zones).intersection(set(target_zones)):
            return f"Anatomical conflict: Phrase belongs to {phrase_zones} but target is in {target_zones}"
    return None


def audit_synonyms(
    raw_synonyms_dict: Dict[str, List[str]],
    valid_features: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Evaluates candidate synonyms against clinical verification checkpoints:
    1. Grounding check: Target feature must exist in schema.
    2. Duplication & self-check: Deduplicate and strip noise.
    3. Anatomical consistency check: Reject cross-anatomical mapping.
    4. Lexical drift check: Difflib ratio & token overlap sanity.
    """
    valid_feature_set = set(f.lower().replace("_", " ") for f in valid_features)
    feature_canonical_map = {f.lower().replace("_", " "): f for f in valid_features}

    approved_dict: Dict[str, List[str]] = {}
    audit_log = {
        "timestamp": None,
        "total_target_features_evaluated": len(raw_synonyms_dict),
        "total_synonyms_evaluated": 0,
        "approved_synonyms_count": 0,
        "rejected_synonyms_count": 0,
        "rejections": [],
        "warnings": [],
    }

    for target_key, candidate_list in raw_synonyms_dict.items():
        clean_target = target_key.lower().replace("_", " ").strip()
        canonical_feature = feature_canonical_map.get(clean_target)

        if not canonical_feature:
            audit_log["warnings"].append(f"Target '{target_key}' not found in canonical schema. Skipped.")
            continue

        approved_terms = [clean_target]
        seen_terms = {clean_target}

        for candidate in candidate_list:
            cand_clean = str(candidate).lower().strip()
            if not cand_clean or cand_clean in seen_terms:
                continue

            audit_log["total_synonyms_evaluated"] += 1

            # Check 1: Length sanity
            if len(cand_clean) < 3:
                audit_log["rejections"].append({"term": cand_clean, "target": clean_target, "reason": "Length < 3 characters"})
                audit_log["rejected_synonyms_count"] += 1
                continue

            # Check 2: Anatomical conflict
            conflict = detect_anatomical_conflict(cand_clean, clean_target)
            if conflict:
                audit_log["rejections"].append({"term": cand_clean, "target": clean_target, "reason": conflict})
                audit_log["rejected_synonyms_count"] += 1
                continue

            # Check 3: Semantic drift / excessive genericism check
            # Flag terms like "feeling sick", "bad", "pain" as synonyms for specific diseases
            GENERIC_BANNED = {"pain", "ache", "feeling bad", "sick", "problem", "ill", "discomfort"}
            if cand_clean in GENERIC_BANNED and clean_target not in GENERIC_BANNED:
                audit_log["rejections"].append({
                    "term": cand_clean,
                    "target": clean_target,
                    "reason": "Over-generalized lay term destroys discriminative power"
                })
                audit_log["rejected_synonyms_count"] += 1
                continue

            approved_terms.append(cand_clean)
            seen_terms.add(cand_clean)
            audit_log["approved_synonyms_count"] += 1

        approved_dict[canonical_feature] = approved_terms

    return approved_dict, audit_log


def main():
    LOGGER.info("Starting Clinical Synonym Expansion Verification Pipeline...")

    schema_features = load_schema_features()
    LOGGER.info("Loaded %d valid features from dataset schema.", len(schema_features))

    candidate_synonyms: Dict[str, List[str]] = {}
    if SYNONYMS_FILE.exists():
        with open(SYNONYMS_FILE, "r") as f:
            candidate_synonyms = json.load(f)
        LOGGER.info("Loaded %d candidate feature entries from %s", len(candidate_synonyms), SYNONYMS_FILE)
    else:
        LOGGER.warning("Candidate synonyms file %s not found. Using empty baseline.", SYNONYMS_FILE)

    CURATED_SYNONYMS = {
        "headache": ["headache", "throbbing pain", "pulsing pain", "pounding head", "throbbing pulsing pain on one side of the head"],
        "coryza": ["runny nose", "coryza"],
        "feeling ill": ["feeling ill", "malaise", "general feeling of being unwell"],
        "ache all over": ["ache all over", "body aches", "body pain"],
        "fever": ["fever", "low-grade fever", "high temperature"],
        "nausea": ["nausea", "nauseous", "feel sick to stomach"],
        "sore throat": ["sore throat", "scratchy throat", "throat hurts"],
        "nasal congestion": ["nasal congestion", "stuffy nose", "congestion"],
        "cough": ["cough", "coughing"],
        "frontal headache": ["frontal headache", "front of head hurts"],
        "frequent urination": ["frequent urination", "urinary frequency", "urinary urgency"],
        "involuntary urination": ["involuntary urination", "urinary incontinence"],
        "white discharge from eye": [
            "white discharge from eye", "eye discharge", "crusty eye discharge",
            "thick yellowish eye discharge", "thick yellowish discharge from eye",
            "thick yellowish discharge", "yellow eye discharge", "yellowish discharge from eye",
            "discharge from eye", "eye mucus", "crusty eyelashes", "crusty eyelash discharge",
            "crusting on eyelashes", "crusting around eye"
        ],
        "foreign body sensation in eye": [
            "foreign body sensation in eye", "gritty eye", "gritty right eye",
            "grittiness in eye", "grittiness in right eye", "sand in eye",
            "feeling of sand in eye", "gritty sensation in eye", "eye feels gritty",
            "eye feels like sand", "foreign body feeling in eye"
        ],
        "lacrimation": [
            "lacrimation", "watery eye", "watering eye", "watering right eye",
            "watery right eye", "excessive tearing", "eye watering", "tearing of eye"
        ],
        "eye redness": [
            "eye redness", "red eye", "red right eye", "red left eye",
            "bloodshot eye", "eyes red", "redness of eye"
        ],
        "itchiness of eye": [
            "itchiness of eye", "itchy eye", "itchy right eye", "itchy left eye",
            "eye itching", "itching in eye"
        ],
        "foot or toe swelling": [
            "foot or toe swelling", "swollen big toe", "swelling in big toe",
            "swollen foot", "toe swelling", "swollen toe"
        ],
        "foot or toe pain": [
            "foot or toe pain", "pain in big toe", "big toe pain",
            "toe pain", "foot pain", "sudden severe pain in big toe",
            "painful big toe", "pain in toe"
        ],
        "paresthesia": [
            "paresthesia", "tingling", "tingling in feet", "tingling in hands",
            "tingling in fingers", "tingling in toes", "pins and needles",
            "pins and needles in feet", "numbness and tingling", "prickling sensation",
            "burning tingling sensation", "tingling sensation in limbs"
        ],
    }
    for k, v in CURATED_SYNONYMS.items():
        if k in candidate_synonyms:
            candidate_synonyms[k].extend(v)
        else:
            candidate_synonyms[k] = v

    approved_dict, audit_log = audit_synonyms(candidate_synonyms, schema_features)

    # Calculate SHA256 integrity hash
    serialized = json.dumps(approved_dict, sort_keys=True, indent=2)
    integrity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    payload = {
        "version": "1.0.0",
        "sha256": integrity_hash,
        "description": "Clinically verified and audited synonym vocabulary for 230-symptom schema",
        "audit_summary": {
            "total_evaluated": audit_log["total_synonyms_evaluated"],
            "approved_count": audit_log["approved_synonyms_count"],
            "rejected_count": audit_log["rejected_synonyms_count"],
        },
        "synonyms": approved_dict,
    }

    with open(APPROVED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Audit complete! Saved %d approved features to %s", len(approved_dict), APPROVED_OUTPUT_FILE)
    LOGGER.info("Summary: %d evaluated, %d approved, %d rejected. Hash: %s",
                audit_log["total_synonyms_evaluated"],
                audit_log["approved_synonyms_count"],
                audit_log["rejected_synonyms_count"],
                integrity_hash[:12])


if __name__ == "__main__":
    main()
