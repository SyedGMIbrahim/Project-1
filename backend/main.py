from __future__ import annotations

import csv
from collections import defaultdict
import difflib
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, build_text_splitter, read_pdf_file, read_text_file

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "symptom_classifier.joblib"

LOGGER = logging.getLogger(__name__)
RAW_DIR = BASE_DIR / "data" / "raw"
PERSIST_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "medical_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
from evidence_engine import (
    DiseaseHallmarkRegistry,
    calculate_evidence_coverage,
    calculate_sparsity_metrics,
    evaluate_decision_gate,
    select_highest_information_gain_question,
)

TOP_K = 6
MENTOR_DATASET_DIR = BASE_DIR / "data" / "mentor_dataset"

SYNONYMS_FILE = BASE_DIR.parent / "scratch" / "generated_synonyms.json"
APPROVED_SYNONYMS_FILE = BASE_DIR / "approved_synonyms_v1.json"

# ── Evidence-Adaptive Clinical Decision Support constants ────────────────────
DATASET_CSV_PATH = BASE_DIR / "data" / "train" / "Diseases_and_Symptoms_dataset.csv"
HALLMARK_FREQ_THRESHOLD = 0.40       # Symptom frequency >= 40% → hallmark for that disease
TRAINING_SPARSITY_MEAN = 5.87        # Mean positive-symptom count in SMOTE-augmented training rows
TRAINING_SPARSITY_STD = 1.67         # Std dev of positive-symptom count in training rows
TOTAL_SCHEMA_FEATURES = 230          # Number of symptom columns in the dataset
SUPPORTED_COSINE_THRESHOLD = 0.70    # Cosine similarity >= 0.70 → SUPPORTED
UNCERTAIN_COSINE_THRESHOLD = 0.58    # 0.58 <= cosine < 0.70 → UNCERTAIN
SUPPORTED_DIFFLIB_THRESHOLD = 0.90   # Difflib ratio > 0.90 → SUPPORTED (fast-path)
ABSTENTION_PROBABILITY_FLOOR = 0.65  # Classifier probability < 0.65 → ABSTAIN
EVIDENCE_COVERAGE_ABSTAIN = 0.30     # Coverage below 30% with sparse input → ABSTAIN
EVIDENCE_COVERAGE_CLARIFY = 0.40     # Coverage below 40% with uncertain symptoms → CLARIFY
MARGIN_TIE_THRESHOLD = 0.04         # Score difference <= 0.04 → margin near-tie

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
    # Eye-specific synonyms — critical for disambiguation against non-eye features
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
    # Peripheral neuropathy — tingling must map to paresthesia, NOT loss of sensation
    # loss of sensation = negative/hypoesthetic; paresthesia = positive/tingling/pins-and-needles
    "paresthesia": [
        "paresthesia", "tingling", "tingling in feet", "tingling in hands",
        "tingling in fingers", "tingling in toes", "pins and needles",
        "pins and needles in feet", "numbness and tingling", "prickling sensation",
        "burning tingling sensation", "tingling sensation in limbs"
    ],
}

FEATURE_SYNONYMS: Dict[str, List[str]] = CURATED_SYNONYMS
try:
    if APPROVED_SYNONYMS_FILE.exists():
        with open(APPROVED_SYNONYMS_FILE, "r") as f:
            payload = json.load(f)
            FEATURE_SYNONYMS = payload.get("synonyms", CURATED_SYNONYMS)
            LOGGER.info("Loaded approved & audited synonyms from %s (version %s)", APPROVED_SYNONYMS_FILE.name, payload.get("version", "1.0"))
    elif SYNONYMS_FILE.exists():
        with open(SYNONYMS_FILE, "r") as f:
            FEATURE_SYNONYMS = json.load(f)
            for k, v in CURATED_SYNONYMS.items():
                if k in FEATURE_SYNONYMS:
                    FEATURE_SYNONYMS[k].extend(v)
                else:
                    FEATURE_SYNONYMS[k] = v
except Exception as e:
    LOGGER.warning(f"Failed to load synonyms: {e}")
    FEATURE_SYNONYMS = CURATED_SYNONYMS


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question about the uploaded healthcare documents")
    file_name: Optional[str] = Field(None, description="Optional specific document file name to scope the query")


class SummarizeRequest(BaseModel):
    file_name: Optional[str] = Field(None, description="Specific document file name to summarize")


class ExtractFromDocsRequest(BaseModel):
    file_name: Optional[str] = Field(None, description="Specific document file name to extract symptoms from")


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]


class UploadResponse(BaseModel):
    message: str
    original_file_name: str
    stored_file_name: str
    file_type: str
    total_chunks_indexed: int
    source_path: str


class DocumentRecord(BaseModel):
    file_name: str
    source_path: Optional[str] = None
    file_type: Optional[str] = None
    upload_date: Optional[str] = None
    file_size: Optional[int] = None
    status: str
    chunk_count: Optional[int] = None


class DiagnoseRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptom names (strings)")
    uncertain_symptoms: Optional[List[str]] = Field(default=None, description="Borderline or near-tie symptoms")
    unsupported_symptoms: Optional[List[str]] = Field(default=None, description="Unmapped or schema-unsupported symptoms")


class DiagnoseResponse(BaseModel):
    predicted: str
    probability: float
    probabilities: Optional[Dict[str, float]] = None
    # ── Evidence-Adaptive Decision Support fields ──
    decision: str = "DIAGNOSE"                           # "DIAGNOSE" | "ABSTAIN" | "CLARIFY"
    sparsity_score: float = 0.0                          # 1 - (active_count / 230); high = sparse
    active_symptom_count: int = 0                        # Number of SUPPORTED symptoms in vector
    evidence_coverage: float = 0.0                       # Weighted hallmark overlap 0.0–1.0
    abstention_reason: Optional[str] = None              # Human-readable reason when ABSTAIN
    clarification_question: Optional[str] = None         # Targeted question when CLARIFY
    symptom_states: Optional[Dict[str, Any]] = None      # {supported, uncertain, unsupported}
    evidence_details: Optional[Dict[str, Any]] = None    # {present_hallmarks, missing_hallmarks, ...}


class ExtractRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Natural language symptom description")


class AppState:
    vector_store: Optional[Chroma] = None
    embeddings: Optional[HuggingFaceEmbeddings] = None
    ollama_client: Optional[ollama.Client] = None
    text_splitter: Optional[Any] = None
    classifier: Optional[Any] = None
    classifier_features: Optional[List[str]] = None
    feature_embeddings: Optional[Any] = None
    embedding_to_feature: Optional[List[str]] = None
    disease_hallmark_profiles: Optional[Dict[str, Dict[str, float]]] = None
    hallmark_registry: Optional[DiseaseHallmarkRegistry] = None


state = AppState()

CLINICAL_QA_SYSTEM_PROMPT = (
    "You are a precision Clinical Document Intelligence AI.\n"
    "Your ONLY source of truth is the provided clinical document context excerpts.\n\n"
    "ABSOLUTE NEGATIVE CONSTRAINTS (STRICT ZERO TOLERANCE FOR HALLUCINATIONS):\n"
    "1. WHOLE-DOCUMENT COVERAGE:\n"
    "   - You MUST survey all sections of the document across all organ systems (General/Constitutional, "
    "Neurological/HEENT, Respiratory, Gastrointestinal, Examination, Follow-up, etc.).\n"
    "   - Do NOT focus on only one organ system or section. List EVERY symptom mentioned across the entire "
    "document (e.g., fatigue, headache, nasal congestion, dry cough, reflux/dyspeptic symptoms, sour taste).\n\n"
    "2. NEVER INVENT 'POSSIBLE CAUSES' OR DIFFERENTIAL DIAGNOSES:\n"
    "   - Under NO circumstances should you list 'possible causes' or differential diagnoses (such as GERD, "
    "functional dyspepsia, IBS, IBD, SIBO, gastritis, celiac, etc.) unless they are explicitly stated as diagnoses in the text.\n"
    "   - Report ONLY the exact 'Working Impression' or 'Assessment' given in the document (e.g., 'Intermittent "
    "dyspeptic/reflux-type symptoms').\n"
    "   - Explicitly state that the document describes findings as nonspecific and does not establish a definitive diagnosis.\n\n"
    "3. NEVER INVENT SPECIFIC MEDICATIONS OR DRUG CLASSES:\n"
    "   - Do NOT name or recommend specific drug classes (such as antacids, H2 blockers, proton pump inhibitors / PPIs, "
    "analgesics, antibiotics) unless explicitly documented in the patient's record.\n"
    "   - If the document specifies 'clinician-directed symptomatic treatment if required', state that exact phrasing "
    "without fabricating drug names.\n\n"
    "4. GROUNDED SEVERITY ASSESSMENT (DISTINGUISHED FROM DIAGNOSIS):\n"
    "   - For each symptom, report its exact documented severity, frequency, and duration:\n"
    "     * Fatigue: gradual and non-disabling for ~3 months; improved with better sleep at follow-up.\n"
    "     * Headache: mild morning headache, ~1–2 episodes/week; less frequent at follow-up.\n"
    "     * Reflux/dyspeptic symptoms: burning epigastric discomfort ~1–2 times/week after late/spicy meals, occasional "
    "nighttime sour taste; reduced after earlier meals.\n"
    "     * Dry cough: associated with recent upper-respiratory symptoms; subsequently resolved.\n"
    "     * Nasal congestion: mild intermittent congestion by history.\n"
    "   - Note the absence of alarm symptoms (no dysphagia, melena, hematemesis, persistent vomiting, or progressive weight loss) "
    "and normal examination findings.\n"
    "   - Clearly label any overall severity judgment as an inferred assessment of the documented symptom pattern (mild and stable "
    "based on descriptions, lack of alarm symptoms, normal exam, and follow-up improvements), NOT as a clinical diagnosis.\n\n"
    "5. DOCUMENTED RECOMMENDATIONS & LIFESTYLE MEASURES:\n"
    "   - State ONLY the documented recommendations: review meal timing and diet, avoid lying down immediately after meals, "
    "monitor symptom frequency, maintain appropriate sleep and physical activity habits, symptom diary, and reassessment if alarm symptoms develop.\n\n"
    "6. SYNTHETIC DATASET / ASSESSMENT NOTICE:\n"
    "   - Explicitly include this notice: 'The document describes these findings as a synthetic dataset / health assessment and "
    "does not establish a definitive diagnosis.'\n\n"
    "REQUIRED OUTPUT FORMAT:\n"
    "Summary:\n"
    "[Concise summary of patient profile, visit reason, and document context]\n\n"
    "Documented Symptoms & Severity:\n"
    "- [Symptom name]: [Exact documented description, frequency, duration, and follow-up status]\n\n"
    "Severity Assessment:\n"
    "[Inferred assessment of the symptom pattern as mild/stable, with explicit distinction that this is a symptom pattern assessment, not a diagnosis]\n\n"
    "Documented Recommendations & Working Impression:\n"
    "- Working Impression: [Exact working impression from record]\n"
    "- Management Plan: [Documented recommendations and lifestyle measures; NO unmentioned drugs]\n\n"
    "Important Notice:\n"
    "[Notice stating findings are a synthetic dataset / health assessment and do not establish a definitive diagnosis]"
)

CLINICAL_SUMMARY_SYSTEM_PROMPT = (
    "You are a precision Clinical Document Summarization AI.\n"
    "Your job is to provide an accurate, comprehensive clinical summary based EXCLUSIVELY on the provided "
    "patient report excerpts.\n\n"
    "ABSOLUTE RULES (ZERO TOLERANCE FOR HALLUCINATIONS):\n"
    "1. WHOLE-DOCUMENT COVERAGE: Extract all symptoms across all organ systems (General/Constitutional, "
    "Neurological/HEENT, Respiratory, Gastrointestinal, Examination, Follow-up, etc.). Do not focus on only one section.\n"
    "2. NO INVENTED CAUSES / DIFFERENTIAL DIAGNOSES: Do NOT introduce possible causes (e.g., GERD, functional dyspepsia, "
    "IBS, IBD, SIBO). Use only the exact documented working impression (e.g., 'Intermittent dyspeptic/reflux-type symptoms').\n"
    "3. NO INVENTED MEDICATIONS: Do NOT recommend specific drugs (e.g., antacids, H2 blockers, PPIs). If the plan states "
    "'clinician-directed symptomatic treatment if required', quote that exact phrasing.\n"
    "4. GROUNDED SEVERITY: Clearly distinguish documented facts (mild headache 1–2/wk, non-disabling fatigue, absence of alarm "
    "symptoms, normal exam, follow-up resolution) from inferred severity (symptoms appear mild/stable).\n"
    "5. SYNTHETIC RECORD NOTICE: Explicitly state that the document describes these findings as a synthetic dataset and does not "
    "establish a definitive diagnosis.\n\n"
    "Format with these sections:\n"
    "- **Summary**: Patient demographics, visit overview, and synthetic dataset context.\n"
    "- **Documented Symptoms & Severity**: Detailed list of ALL documented symptoms (Fatigue, Headache, Reflux/Dyspeptic symptoms, "
    "Sour taste, Dry cough, Nasal congestion) with documented frequency, duration, and follow-up changes.\n"
    "- **Severity Assessment**: Inferred assessment of the documented symptom pattern (mild and stable).\n"
    "- **Documented Recommendations & Impression**: Exact working impression, lifestyle measures, and follow-up plan.\n"
    "- **Important Notice**: Synthetic dataset / health assessment notice without definitive diagnosis.\n"
)


def build_prompt(query: str, context_chunks: List[str], doc_name: Optional[str] = None) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    doc_header = f"Report Source: {doc_name}\n\n" if doc_name else ""
    return (
        f"{doc_header}Document Context Excerpts (Chronologically Sampled Across All Pages):\n"
        f"\"\"\"\n{context_block}\n\"\"\"\n\n"
        f"User Question / Input:\n\"{query}\"\n\n"
        "Instructions for Response:\n"
        "1. Survey the ENTIRE document context provided above across all body systems (General/Constitutional, Neurological, "
        "HEENT, Respiratory, GI, Exam, Follow-up). Do not focus only on one section.\n"
        "2. List EVERY documented symptom (e.g., fatigue, headache, nasal congestion, dry cough, reflux/dyspeptic symptoms, sour taste) "
        "and describe its documented severity, frequency, duration, and follow-up status.\n"
        "3. Under NO circumstances suggest or invent 'possible causes' or differential diagnoses (such as GERD, functional "
        "dyspepsia, IBS, IBD, SIBO). Report only the exact working impression.\n"
        "4. Do NOT recommend specific unmentioned medications (such as antacids, H2 blockers, PPIs). If the record states "
        "'clinician-directed symptomatic treatment if required', state that exact phrasing.\n"
        "5. Clearly distinguish documented facts from inferred severity (label overall severity as an inferred assessment of "
        "the documented symptom pattern, not a diagnosis).\n"
        "6. If the document is a synthetic assessment / dataset, state that explicitly.\n\n"
        "Clinical Answer:"
    )


def build_summary_prompt(context_chunks: List[str], doc_name: Optional[str] = None) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    doc_header = f"Report Source: {doc_name}\n\n" if doc_name else ""
    return (
        f"{doc_header}Document Context Excerpts (Chronologically Sampled Across All Pages):\n"
        f"\"\"\"\n{context_block}\n\"\"\"\n\n"
        "Instructions:\n"
        "- Provide a comprehensive summary covering the ENTIRE document across all organ systems.\n"
        "- Extract and list EVERY documented symptom across all body systems (e.g., fatigue, headache, nasal congestion, "
        "cough, reflux/dyspeptic symptoms, sour taste) with its documented severity, frequency, and follow-up status.\n"
        "- Do NOT invent possible causes or differential diagnoses (no GERD, IBS, IBD, SIBO, etc.). Use only the documented "
        "working impression.\n"
        "- Do NOT invent specific medications (no antacids, H2 blockers, PPIs). State only clinician-directed treatment if documented.\n"
        "- Clearly label overall severity as an inferred assessment of the symptom pattern (mild/stable).\n"
        "- Note that the document is a synthetic health assessment and does not establish a definitive diagnosis.\n\n"
        "Clinical Summary:"
    )


def resolve_document_filter(file_name: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Resolves a ChromaDB metadata filter to guarantee queries are scoped to the target clinical document.
    Prevents cross-contamination from background datasets like mtsamples.csv.
    """
    if file_name and file_name.strip():
        target = file_name.strip()
        # Check if target matches exactly in RAW_DIR
        if (RAW_DIR / target).exists():
            return {"file_name": target}, target
        # Check if target matches by stem or prefix in RAW_DIR
        for p in RAW_DIR.glob("*"):
            if p.is_file() and (p.name == target or p.name.startswith(Path(target).stem)):
                return {"file_name": p.name}, p.name
        # Fallback to the exact string provided
        return {"file_name": target}, target

    # Default to the most recently modified uploaded file in RAW_DIR
    raw_files = [p for p in RAW_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}]
    if raw_files:
        latest = max(raw_files, key=lambda f: f.stat().st_mtime)
        LOGGER.info("No file_name specified; defaulting to latest uploaded file: %s", latest.name)
        return {"file_name": latest.name}, latest.name

    return None, None


def map_extracted_to_features_3state(
    extracted_list: List[str], valid_features: List[str]
) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
    """
    Three-State Epistemic Symptom Classifier:
    - SUPPORTED: Validated feature entering model representation (cosine >= 0.70 or difflib > 0.90, margin > 0.04).
    - UNCERTAIN: Borderline feature (0.58 <= cosine < 0.70 OR margin tie <= 0.04) retained for evidence gating / clarification.
    - UNSUPPORTED: Described by patient but lacks representation in schema (< 0.58 or anatomical conflict).
    """
    supported_symptoms: List[str] = []
    uncertain_symptoms: List[str] = []
    unsupported_symptoms: List[str] = []
    mapping_details: List[Dict[str, Any]] = []

    feature_lower = [f.lower().replace("_", " ") for f in valid_features]

    for symptom in extracted_list:
        symptom_clean = str(symptom).lower().strip()
        if len(symptom_clean) < 3:
            unsupported_symptoms.append(symptom_clean)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "UNSUPPORTED",
                "matched_feature": None,
                "score": 0.0,
                "reason": "Character length < 3",
            })
            continue

        LOGGER.info("Mapping extracted symptom: '%s'", symptom_clean)

        # 1. Fast-path: Exact or near-exact difflib string match
        best_ratio = 0.0
        best_orig = None
        for orig, lowered in zip(valid_features, feature_lower):
            ratio = difflib.SequenceMatcher(None, symptom_clean, lowered).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_orig = orig

        if best_ratio > SUPPORTED_DIFFLIB_THRESHOLD and best_orig:
            LOGGER.info("  -> Selected via fast-path (ratio=%.3f): '%s'", best_ratio, best_orig)
            if best_orig not in supported_symptoms:
                supported_symptoms.append(best_orig)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "SUPPORTED",
                "matched_feature": best_orig,
                "score": round(best_ratio, 3),
                "method": "difflib_fast_path",
                "reason": f"Exact/near-exact match (ratio {round(best_ratio, 3)})",
            })
            continue

        # 2. Semantic Embedding Similarity Match
        if state.embeddings is None or state.feature_embeddings is None or state.embedding_to_feature is None:
            LOGGER.warning("  -> Embeddings uninitialized, tagging as unmapped")
            unsupported_symptoms.append(symptom_clean)
            continue

        phrase_emb = np.array(state.embeddings.embed_query(symptom_clean)).reshape(1, -1)
        sims = cosine_similarity(phrase_emb, state.feature_embeddings)[0]
        top_indices = sims.argsort()[-5:][::-1]

        # Anatomical conflict guards and Taste gap guard
        EYE_TERMS = {"eye", "ocular", "eyelash", "eyelid", "conjunctiv", "optic", "pupil", "iris", "gritty", "watery", "lacrimation"}
        NASAL_TERMS = {"nasal", "nose", "sinus", "smell"}
        TASTE_TERMS = {"taste", "ageusia", "dysgeusia", "flavor"}

        GENITAL_FEATURES = {"vaginal", "penile", "genital", "vulvar", "scrotum", "testes"}
        EYE_FEATURES = {"eye", "eyelid", "conjunctiva", "vision"}
        MOUTH_PAIN_FEATURES = {"mouth pain", "oral pain", "pain in gums", "toothache"}

        phrase_has_eye = any(t in symptom_clean for t in EYE_TERMS)
        phrase_has_nasal = any(t in symptom_clean for t in NASAL_TERMS)
        phrase_has_taste = any(t in symptom_clean for t in TASTE_TERMS)

        def is_conflict(phrase, candidate_lower):
            if phrase_has_eye and any(g in candidate_lower for g in GENITAL_FEATURES): return True
            if phrase_has_nasal and any(g in candidate_lower for g in EYE_FEATURES.union(GENITAL_FEATURES)): return True
            if phrase_has_taste and any(g in candidate_lower for g in MOUTH_PAIN_FEATURES): return True
            return False

        valid_candidates = []
        seen_features = set()

        for idx in top_indices:
            candidate_orig = state.embedding_to_feature[idx]
            candidate_lower = candidate_orig.lower()
            score = float(sims[idx])

            if score < UNCERTAIN_COSINE_THRESHOLD:
                continue

            if is_conflict(symptom_clean, candidate_lower):
                LOGGER.warning("  -> Rejected '%s' for phrase '%s' (anatomical/context conflict).", candidate_orig, symptom_clean)
                continue

            if candidate_orig not in seen_features:
                valid_candidates.append((candidate_orig, score))
                seen_features.add(candidate_orig)

        if not valid_candidates:
            LOGGER.info("  -> UNSUPPORTED (no candidates >= %.2f or all conflicted)", UNCERTAIN_COSINE_THRESHOLD)
            unsupported_symptoms.append(symptom_clean)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "UNSUPPORTED",
                "matched_feature": None,
                "score": float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0,
                "reason": "Described in text but no schema feature meets similarity floor (genuine coverage gap)",
            })
            continue

        best_orig, best_score = valid_candidates[0]

        # Evaluate margin near-tie
        is_margin_tie = False
        runner_up_orig, runner_up_score = None, 0.0
        if len(valid_candidates) > 1:
            runner_up_orig, runner_up_score = valid_candidates[1]
            if (best_score - runner_up_score) <= MARGIN_TIE_THRESHOLD:
                is_margin_tie = True

        if is_margin_tie:
            LOGGER.info("  -> UNCERTAIN (margin near-tie: '%s' (%.3f) vs '%s' (%.3f))", best_orig, best_score, runner_up_orig, runner_up_score)
            uncertain_symptoms.append(best_orig)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "UNCERTAIN",
                "matched_feature": best_orig,
                "runner_up": runner_up_orig,
                "score": round(best_score, 3),
                "margin": round(best_score - runner_up_score, 3),
                "reason": f"Near-tie competition between '{best_orig}' and '{runner_up_orig}' (margin <= {MARGIN_TIE_THRESHOLD})",
            })
        elif best_score < SUPPORTED_COSINE_THRESHOLD:
            LOGGER.info("  -> UNCERTAIN (in uncertainty corridor %.2f <= %.3f < %.2f: '%s')",
                        UNCERTAIN_COSINE_THRESHOLD, best_score, SUPPORTED_COSINE_THRESHOLD, best_orig)
            uncertain_symptoms.append(best_orig)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "UNCERTAIN",
                "matched_feature": best_orig,
                "score": round(best_score, 3),
                "reason": f"Moderate semantic confidence in uncertainty band [{UNCERTAIN_COSINE_THRESHOLD}, {SUPPORTED_COSINE_THRESHOLD})",
            })
        else:
            LOGGER.info("  -> SUPPORTED via semantics: '%s' (score: %.3f)", best_orig, best_score)
            if best_orig not in supported_symptoms:
                supported_symptoms.append(best_orig)
            mapping_details.append({
                "raw_text": symptom_clean,
                "state": "SUPPORTED",
                "matched_feature": best_orig,
                "score": round(best_score, 3),
                "reason": f"High-confidence semantic alignment (score >= {SUPPORTED_COSINE_THRESHOLD})",
            })

    if unsupported_symptoms:
        LOGGER.warning("Symptoms described but not supported by current symptom schema: %s", ", ".join(unsupported_symptoms))

    return supported_symptoms, uncertain_symptoms, unsupported_symptoms, mapping_details


def map_extracted_to_features(extracted_list: List[str], valid_features: List[str]) -> List[str]:
    """Backward-compatible mapping function returning only SUPPORTED symptoms."""
    supported, _, _, _ = map_extracted_to_features_3state(extracted_list, valid_features)
    return supported


SYMPTOM_EXTRACTION_PROMPT_TEMPLATE = """
You are a thorough medical symptom extractor. Your task is to read the text below and extract EVERY SINGLE symptom or physical complaint mentioned, without exception.

Rules:
1. Extract ALL symptoms — do not stop after the first one. Read the entire text and list every symptom.
2. Preserve anatomical and contextual specificity: write "swelling in big toe" not "swelling", "crusty eye discharge" not "discharge", "gritty eye sensation" not "eye discomfort".
3. Treat each distinct complaint as a separate item (e.g. redness, itchiness, discharge, gritty sensation, watering are all separate entries).
4. Do NOT summarize or group symptoms together.
5. Respond ONLY with a valid JSON object: {{"symptoms": ["symptom 1", "symptom 2", ...]}}

Example for a text describing 5 symptoms: {{"symptoms": ["runny nose", "congestion", "scratchy throat", "coughing", "sneezing"]}}

Text: {source_text}
"""


def extract_symptoms_3state_with_llm(source_text: str, feature_columns: List[str]) -> Dict[str, Any]:
    """Extracts symptoms and classifies them into the three epistemic states."""
    if state.ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client is not ready")

    prompt = SYMPTOM_EXTRACTION_PROMPT_TEMPLATE.format(source_text=source_text)

    try:
        response = state.ollama_client.chat(
            model=DEFAULT_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a thorough medical symptom extractor. You MUST extract every symptom mentioned in the text without exception. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 512,
            },
        )

        parsed = json.loads(response["message"]["content"])
        extracted_list = parsed.get("symptoms", [])

        supported, uncertain, unsupported, details = map_extracted_to_features_3state(extracted_list, feature_columns)
        return {
            "symptoms": supported,
            "uncertain": uncertain,
            "unsupported": unsupported,
            "details": details,
        }

    except Exception as e:
        LOGGER.exception("Extraction error: %s", e)
        return {"symptoms": [], "uncertain": [], "unsupported": [], "details": []}


def extract_symptoms_with_llm(source_text: str, feature_columns: List[str]) -> List[str]:
    """Backward-compatible extraction function returning list of supported symptoms."""
    result = extract_symptoms_3state_with_llm(source_text, feature_columns)
    return result.get("symptoms", [])


def ensure_upload_directory() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def validate_upload_file(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are accepted")
    return suffix


def sanitize_file_name(file_name: str) -> str:
    safe_name = Path(file_name).name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Uploaded file name is invalid")
    return safe_name


def list_candidate_files() -> Dict[str, Path]:
    indexed_directories = [RAW_DIR, MENTOR_DATASET_DIR]
    candidate_files: Dict[str, Path] = {}

    for directory in indexed_directories:
        if not directory.exists():
            continue

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt"}:
                candidate_files[str(file_path.resolve())] = file_path

    return candidate_files


def format_datetime(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def get_indexed_document_records(limit: int = 100, offset: int = 0) -> List[DocumentRecord]:
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Backend resources are not ready")

    chroma_sources: Dict[str, int] = {}
    chroma_file_types: Dict[str, str] = {}

    # Clamp limit to a safe maximum to protect the underlying SQLite implementation
    MAX_LIMIT = 1000
    if limit is None or limit <= 0:
        limit = 100
    if limit > MAX_LIMIT:
        LOGGER.warning("Requested limit %d exceeds max %d, clamping to %d", limit, MAX_LIMIT, MAX_LIMIT)
        limit = MAX_LIMIT

    try:
        # Limit the number of metadata rows retrieved to avoid SQLite 'too many SQL variables' errors
        collection_payload = state.vector_store._collection.get(include=["metadatas"], limit=limit, offset=offset)  # type: ignore[attr-defined]
    except Exception as exc:
        LOGGER.exception("Failed to query Chroma metadata")
        raise HTTPException(status_code=500, detail="Failed to inspect indexed documents") from exc

    metadatas = collection_payload.get("metadatas") or []
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue

        source_path = metadata.get("source")
        if not isinstance(source_path, str) or not source_path:
            continue

        chroma_sources[source_path] = chroma_sources.get(source_path, 0) + 1
        if source_path not in chroma_file_types:
            file_type = metadata.get("file_type")
            chroma_file_types[source_path] = file_type if isinstance(file_type, str) else Path(source_path).suffix.lstrip(".")

    filesystem_files = list_candidate_files()
    records: List[DocumentRecord] = []

    for source_path, chunk_count in sorted(chroma_sources.items(), key=lambda item: item[0].lower()):
        file_path = filesystem_files.get(str(Path(source_path).resolve())) or Path(source_path)
        exists = file_path.exists()
        stat_result = file_path.stat() if exists else None

        records.append(
            DocumentRecord(
                file_name=file_path.name,
                source_path=str(file_path) if exists or file_path.exists() else source_path,
                file_type=chroma_file_types.get(source_path) or file_path.suffix.lstrip("."),
                upload_date=format_datetime(stat_result.st_mtime if stat_result else None),
                file_size=stat_result.st_size if stat_result else None,
                status="indexed" if chunk_count > 0 else "pending",
                chunk_count=chunk_count,
            )
        )

    return records


def build_documents_from_upload(file_path: Path, file_type: str) -> List[Document]:
    if file_type == ".pdf":
        page_texts = read_pdf_file(file_path)
        source_documents = [
            Document(
                page_content=page_text.strip(),
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page": page_number,
                    "page_number": page_number,
                },
            )
        for page_number, page_text in page_texts
            if page_text.strip()
        ]
    else:
        text = read_text_file(file_path).strip()
        source_documents = []
        if text:
            source_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "txt",
                        "page": None,
                        "page_number": None,
                    },
                )
            )

    if not source_documents:
        return []

    splitter = state.text_splitter or build_text_splitter(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    chunked_documents = splitter.split_documents(source_documents)

    for chunk_index, document in enumerate(chunked_documents, start=1):
        document.metadata["chunk_index"] = chunk_index
        document.metadata["chunk_size"] = len(document.page_content)

    return chunked_documents


def persist_uploaded_file(upload_file: UploadFile, file_name: str) -> Tuple[Path, str, int, str]:
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Backend resources are not ready")

    ensure_upload_directory()
    suffix = validate_upload_file(file_name)
    safe_name = sanitize_file_name(file_name)

    payload = upload_file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    digest = hashlib.sha1(payload).hexdigest()[:10]
    target_name = f"{Path(safe_name).stem}-{digest}{suffix}"
    target_path = RAW_DIR / target_name
    target_path.write_bytes(payload)

    chunked_documents = build_documents_from_upload(target_path, suffix)
    if not chunked_documents:
        raise HTTPException(status_code=400, detail="Uploaded document did not contain any indexable text")

    state.vector_store.add_documents(chunked_documents)
    if hasattr(state.vector_store, "persist"):
        state.vector_store.persist()

    return target_path, safe_name, len(chunked_documents), suffix.lstrip(".")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info(f"Active extraction prompt template: {SYMPTOM_EXTRACTION_PROMPT_TEMPLATE}")
    LOGGER.info("Loading local embeddings, Chroma index, and Ollama client")

    state.embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    state.vector_store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=state.embeddings,
    )
    state.ollama_client = ollama.Client(host=DEFAULT_OLLAMA_HOST)
    state.text_splitter = build_text_splitter(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    # Load ML classifier if available
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Symptom classifier model not found at {CLASSIFIER_PATH}")

    try:
        cls_payload = joblib.load(CLASSIFIER_PATH)
        if isinstance(cls_payload, dict) and "model" in cls_payload and "features" in cls_payload:
            state.classifier = cls_payload["model"]
            state.classifier_features = list(cls_payload["features"])
            LOGGER.info("Loaded symptom classifier with %d features", len(state.classifier_features))
            
            # Precompute embeddings for all features and their synonyms
            LOGGER.info("Computing semantic embeddings for feature mapping...")
            embedding_strings = []
            embedding_feature_map = []
            
            for f in state.classifier_features:
                clean_f = f.lower().replace("_", " ")
                synonyms = FEATURE_SYNONYMS.get(clean_f, [clean_f])
                for syn in synonyms:
                    embedding_strings.append(syn)
                    embedding_feature_map.append(f)
                
            state.feature_embeddings = np.array(state.embeddings.embed_documents(embedding_strings))
            state.embedding_to_feature = embedding_feature_map
            LOGGER.info("Successfully cached %d feature embeddings for %d features.", len(embedding_strings), len(state.classifier_features))
        else:
            # assume direct model object
            state.classifier = cls_payload
            state.classifier_features = None
            LOGGER.info("Loaded symptom classifier (features unknown)")
    except Exception as e:
        LOGGER.exception("Failed to load symptom classifier")
        raise RuntimeError(f"Failed to load symptom classifier: {e}") from e

    # Initialize Disease Hallmark Registry for Evidence-Adaptive Decision Support
    try:
        LOGGER.info("Initializing Disease Hallmark Registry from dataset...")
        state.hallmark_registry = DiseaseHallmarkRegistry(DATASET_CSV_PATH)
        LOGGER.info("Hallmark Registry initialized for %d diseases.", len(state.hallmark_registry.disease_counts))
    except Exception as e:
        LOGGER.warning("Could not initialize hallmark registry at startup: %s", e)

    yield

    state.vector_store = None
    state.embeddings = None
    state.ollama_client = None
    state.text_splitter = None
    state.classifier = None
    state.classifier_features = None
    state.hallmark_registry = None


app = FastAPI(title="Healthcare RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents", response_model=List[DocumentRecord])
def list_documents(limit: int = 100, offset: int = 0) -> List[DocumentRecord]:
    """List indexed documents with a safe limit and optional offset to avoid DB overload.

    Query params:
    - limit: number of metadata rows to retrieve (default 100, max 1000)
    - offset: offset into the metadata rows for pagination
    """
    return get_indexed_document_records(limit=limit, offset=offset)


@app.get("/api/symptoms")
def list_symptoms() -> List[str]:
    if state.classifier_features is None:
        raise HTTPException(status_code=404, detail="No symptom feature list available")
    return state.classifier_features


@app.post("/api/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    if state.classifier is None or state.classifier_features is None:
        raise HTTPException(status_code=503, detail="Classifier is not loaded")

    if state.hallmark_registry is None:
        state.hallmark_registry = DiseaseHallmarkRegistry(DATASET_CSV_PATH)

    # Build feature vector in the trained order
    feature_names = state.classifier_features
    selected = set([s.lower().strip() for s in req.symptoms or []])

    x = [1 if fname.lower().strip() in selected else 0 for fname in feature_names]

    try:
        import numpy as np

        arr = np.array(x).reshape(1, -1)
        if hasattr(state.classifier, "predict_proba"):
            proba = state.classifier.predict_proba(arr)[0]
            classes = list(state.classifier.classes_)
            sorted_predictions = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)

            top_pred_class, top_pred_prob = sorted_predictions[0]
            pred = str(top_pred_class)
            probability = float(top_pred_prob)

            # Top 3 predicted diseases and their exact probability percentages
            top_3 = sorted_predictions[:3]
            probs = {str(c): float(p) for c, p in top_3}
            top_candidates = [(str(c), float(p)) for c, p in sorted_predictions[:5]]
        else:
            pred = state.classifier.predict(arr)[0]
            probability = 1.0
            probs = {str(pred): 1.0}
            top_candidates = [(str(pred), 1.0)]

    except Exception as exc:
        LOGGER.exception("Diagnosis failed")
        raise HTTPException(status_code=500, detail="Failed to run diagnosis") from exc

    # Evaluate Evidence-Adaptive Decision Gate: T = f(S, A, C, N)
    gate_result = evaluate_decision_gate(
        top_predictions=top_candidates,
        supported_symptoms=list(selected),
        uncertain_symptoms=req.uncertain_symptoms or [],
        unsupported_symptoms=req.unsupported_symptoms or [],
        hallmark_registry=state.hallmark_registry,
    )

    return DiagnoseResponse(
        predicted=gate_result["predicted"],
        probability=probability,
        probabilities=probs,
        decision=gate_result["decision"],
        sparsity_score=gate_result["sparsity_info"]["sparsity_score"],
        active_symptom_count=gate_result["sparsity_info"]["active_symptom_count"],
        evidence_coverage=gate_result["evidence_coverage"],
        abstention_reason=gate_result["abstention_reason"],
        clarification_question=gate_result["clarification_question"],
        symptom_states={
            "supported": list(selected),
            "uncertain": req.uncertain_symptoms or [],
            "unsupported": req.unsupported_symptoms or [],
        },
        evidence_details=gate_result["evidence_details"],
    )


@app.post("/api/extract-symptoms")
async def extract_symptoms(request: dict):
    text = request.get("text", "") or request.get("description", "")
    feature_columns = state.classifier_features or []
    if not text:
        return {"symptoms": [], "uncertain": [], "unsupported": [], "details": []}

    extraction_result = extract_symptoms_3state_with_llm(text, feature_columns)
    return extraction_result


@app.post("/api/extract-from-docs")
def extract_symptoms_from_docs(payload: Optional[ExtractFromDocsRequest] = None) -> Dict[str, List[str]]:
    if state.classifier_features is None:
        raise HTTPException(status_code=404, detail="No symptom feature list available")
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready")

    requested_file = payload.file_name if payload else None
    filter_dict, target_doc = resolve_document_filter(requested_file)

    query_text = "patient symptoms, chief complaint, physical signs, clinical presentation, abnormal lab findings, glucose, pain, severity"
    try:
        if filter_dict:
            LOGGER.info("Extracting symptoms scoped to document: %s", target_doc)
            retrieved_documents = state.vector_store.similarity_search(query_text, k=8, filter=filter_dict)
            if not retrieved_documents:
                LOGGER.warning("No chunks found with filter %s; attempting search across all uploaded docs", filter_dict)
                retrieved_documents = state.vector_store.similarity_search(query_text, k=6)
        else:
            retrieved_documents = state.vector_store.similarity_search(query_text, k=6)
    except Exception as exc:
        LOGGER.exception("Vector search failed for extract-from-docs: %s", exc)
        try:
            retrieved_documents = state.vector_store.similarity_search(query_text, k=6)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to retrieve document context") from exc

    if not retrieved_documents:
        return {"symptoms": []}

    context_block = "\n\n".join([doc.page_content for doc in retrieved_documents if doc.page_content])
    if not context_block.strip():
        return {"symptoms": []}

    feature_columns = state.classifier_features
    final_symptoms = extract_symptoms_with_llm(source_text=context_block, feature_columns=feature_columns)
    return {"symptoms": final_symptoms}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")

    try:
        saved_path, original_file_name, total_chunks_indexed, file_type = persist_uploaded_file(file, file.filename)
    finally:
        await file.close()

    return UploadResponse(
        message="File uploaded and indexed successfully",
        original_file_name=original_file_name,
        stored_file_name=saved_path.name,
        file_type=file_type,
        total_chunks_indexed=total_chunks_indexed,
        source_path=str(saved_path),
    )


def is_broad_or_summary_query(text: str) -> bool:
    patterns = [
        r"\bsummar[a-z]*\b",             # summarize, summarise, summary, summarising, summarization, etc.
        r"\boverview\b",
        r"\bbrief\b",
        r"\bcase\s+report\b",
        r"\bentire\s+doc[a-z]*\b",
        r"\bwhole\s+doc[a-z]*\b",
        r"\ball\s+symptom[s]?\b",
        r"\bsuggest\s+symptom[s]?\b",
        r"\bsymptoms?\s+and\s+severity\b",
        r"\bhealth\s+assessment\b",
        r"\bcomprehensive\b",
        r"\bdocument\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def retrieve_comprehensive_document_context(
    query_text: str,
    filter_dict: Optional[Dict[str, Any]],
    is_summary_or_broad: bool = False,
    max_chunks: int = 20,
) -> List[Document]:
    """
    Retrieves chronological, whole-document context across all body systems.
    Prevents single-topic clustering (e.g. only GI) by sampling evenly across all pages
    from page 1 to the final page and merging with targeted query matches.
    """
    if filter_dict and hasattr(state.vector_store, "_collection"):
        try:
            matched = state.vector_store._collection.get(where=filter_dict, include=["documents", "metadatas"])
            docs_text = matched.get("documents") or []
            metas = matched.get("metadatas") or []

            if docs_text:
                combined = []
                for doc_text, meta in zip(docs_text, metas):
                    idx = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
                    combined.append((idx, Document(page_content=doc_text, metadata=meta or {})))
                # Sort in strict chronological order by chunk_index
                combined.sort(key=lambda item: item[0])
                all_docs = [doc for _, doc in combined]
                total_docs = len(all_docs)

                if is_summary_or_broad:
                    if total_docs <= max_chunks:
                        return all_docs

                    # Evenly sample indices across the entire document from 0 to total_docs - 1
                    step = (total_docs - 1) / (max_chunks - 1)
                    selected_indices = set(int(round(i * step)) for i in range(max_chunks))
                    # Ensure first 2 and last 2 chunks are always included
                    selected_indices.update([0, 1, total_docs - 2, total_docs - 1])

                    sampled_docs = [all_docs[i] for i in sorted(selected_indices) if 0 <= i < total_docs]

                    # Augment with any specific semantic matches from query_text
                    if query_text:
                        try:
                            sim_matches = state.vector_store.similarity_search(query_text, k=6, filter=filter_dict)
                            existing_texts = set(d.page_content for d in sampled_docs)
                            for doc in sim_matches:
                                if doc.page_content not in existing_texts:
                                    sampled_docs.append(doc)
                                    existing_texts.add(doc.page_content)
                        except Exception as e:
                            LOGGER.warning("Similarity augmentation failed: %s", e)

                    sampled_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0) if isinstance(d.metadata, dict) else 0)
                    return sampled_docs
                else:
                    # Specific query: get top similarity search matches
                    try:
                        sim_matches = state.vector_store.similarity_search(query_text, k=TOP_K, filter=filter_dict)
                    except Exception:
                        sim_matches = []

                    if sim_matches:
                        # Append the last 2 chunks (Assessment/Plan/Follow-up) so clinician conclusions are in context
                        last_chunks = all_docs[-2:]
                        existing_texts = set(d.page_content for d in sim_matches)
                        for doc in last_chunks:
                            if doc.page_content not in existing_texts:
                                sim_matches.append(doc)
                                existing_texts.add(doc.page_content)
                        sim_matches.sort(key=lambda d: d.metadata.get("chunk_index", 0) if isinstance(d.metadata, dict) else 0)
                        return sim_matches
        except Exception as exc:
            LOGGER.warning("Direct collection retrieval failed: %s; falling back to similarity search", exc)

    # Fallback to similarity search
    search_k = max_chunks if is_summary_or_broad else TOP_K
    try:
        if filter_dict:
            res = state.vector_store.similarity_search(query_text, k=search_k, filter=filter_dict)
            if res:
                return res
        return state.vector_store.similarity_search(query_text, k=search_k)
    except Exception as exc:
        LOGGER.exception("Vector search fallback failed: %s", exc)
        return []


@app.post("/api/summarize", response_model=QueryResponse)
def summarize_document(payload: Optional[SummarizeRequest] = None) -> QueryResponse:
    if state.vector_store is None or state.ollama_client is None:
        raise HTTPException(status_code=503, detail="Backend resources are not ready")

    requested_file = payload.file_name if payload else None
    filter_dict, target_doc = resolve_document_filter(requested_file)

    retrieved_documents = retrieve_comprehensive_document_context(
        query_text="chief complaint, history of present illness, diagnosis, physical findings, lab results, clinical assessment, treatment plan",
        filter_dict=filter_dict,
        is_summary_or_broad=True,
        max_chunks=20,
    )

    if not retrieved_documents:
        raise HTTPException(status_code=404, detail="No relevant context found in the local documents to summarize")

    context_chunks = [document.page_content for document in retrieved_documents]
    prompt = build_summary_prompt(context_chunks, doc_name=target_doc)

    try:
        ollama_response = state.ollama_client.chat(
            model=DEFAULT_OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": CLINICAL_SUMMARY_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 1200},
        )
    except Exception as exc:
        LOGGER.exception("Ollama summary generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate clinical summary with local Ollama model") from exc

    answer_text = ollama_response["message"]["content"].strip()
    source_documents = [
        SourceDocument(content=document.page_content, metadata=dict(document.metadata or {}))
        for document in retrieved_documents
    ]

    return QueryResponse(answer=answer_text, source_documents=source_documents)


@app.post("/api/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    if state.vector_store is None or state.ollama_client is None:
        raise HTTPException(status_code=503, detail="Backend resources are not ready")

    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    filter_dict, target_doc = resolve_document_filter(payload.file_name)

    is_summary = is_broad_or_summary_query(query_text)
    LOGGER.info("Query: '%s' (is_summary_or_broad=%s, target_doc=%s)", query_text, is_summary, target_doc)

    retrieved_documents = retrieve_comprehensive_document_context(
        query_text=query_text,
        filter_dict=filter_dict,
        is_summary_or_broad=is_summary,
        max_chunks=20,
    )

    if not retrieved_documents:
        raise HTTPException(status_code=404, detail="No relevant context found in the local documents")

    context_chunks = [document.page_content for document in retrieved_documents]
    prompt = build_prompt(query_text, context_chunks, doc_name=target_doc)

    try:
        ollama_response = state.ollama_client.chat(
            model=DEFAULT_OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": CLINICAL_QA_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 1200},
        )
    except Exception as exc:
        LOGGER.exception("Ollama generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate an answer with the local Ollama model") from exc

    answer_text = ollama_response["message"]["content"].strip()
    source_documents = [
        SourceDocument(content=document.page_content, metadata=dict(document.metadata or {}))
        for document in retrieved_documents
    ]

    return QueryResponse(answer=answer_text, source_documents=source_documents)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
