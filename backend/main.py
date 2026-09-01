from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K = 3
MENTOR_DATASET_DIR = BASE_DIR / "data" / "mentor_dataset"

SYNONYMS_FILE = BASE_DIR / "data" / "synonyms.json"
FALLBACK_SYNONYMS_FILE = BASE_DIR.parent / "scratch" / "generated_synonyms.json"

CURATED_SYNONYMS = {
    "headache": ["headache", "throbbing pain", "pulsing pain", "pounding head", "throbbing pulsing pain on one side of the head", "migraine", "head pain"],
    "coryza": ["runny nose", "coryza", "rhinorrhea", "running nose"],
    "feeling ill": ["feeling ill", "malaise", "general feeling of being unwell", "feeling unwell", "sluggish"],
    "ache all over": ["ache all over", "body aches", "body pain", "generalized body aches", "muscle aches all over"],
    "fever": ["fever", "low-grade fever", "high temperature", "pyrexia", "elevated temperature", "febrile"],
    "nausea": ["nausea", "nauseous", "feel sick to stomach", "feeling queasy", "sick to my stomach"],
    "sore throat": ["sore throat", "scratchy throat", "throat hurts", "throat pain", "raw throat", "pharyngitis"],
    "nasal congestion": ["nasal congestion", "stuffy nose", "congestion", "blocked nose", "sinus congestion"],
    "cough": ["cough", "coughing", "dry cough", "productive cough", "persistent cough"],
    "frontal headache": ["frontal headache", "front of head hurts", "sinus headache", "forehead headache", "facial pressure"],
    "frequent urination": ["frequent urination", "urinary frequency", "urinary urgency", "peeing all the time", "need to pee often", "urinating frequently", "polyuria"],
    "involuntary urination": ["involuntary urination", "urinary incontinence", "leaking urine", "cannot hold urine", "loss of bladder control"],
    "painful urination": ["painful urination", "burning pain when i pee", "burning urination", "dysuria", "hurts to pee", "pain when urinating", "burning sensation when peeing"],
    "lower abdominal pain": ["lower abdominal pain", "lower stomach hurts", "pelvic pain", "lower belly pain", "suprapubic pain", "cramping in lower abdomen"],
    # Eye-specific synonyms — critical for disambiguation against non-eye features
    "white discharge from eye": [
        "white discharge from eye", "eye discharge", "crusty eye discharge",
        "thick yellowish eye discharge", "thick yellowish discharge from eye",
        "thick yellowish discharge", "yellow eye discharge", "yellowish discharge from eye",
        "discharge from eye", "eye mucus", "crusty eyelashes", "crusty eyelash discharge",
        "crusting on eyelashes", "crusting around eye", "pus from eye"
    ],
    "foreign body sensation in eye": [
        "foreign body sensation in eye", "gritty eye", "gritty right eye", "gritty left eye",
        "grittiness in eye", "grittiness in right eye", "grittiness in left eye", "sand in eye",
        "feeling of sand in eye", "gritty sensation in eye", "eye feels gritty",
        "eye feels like sand", "foreign body feeling in eye", "gritty sensation"
    ],
    "lacrimation": [
        "lacrimation", "watery eye", "watering eye", "watering right eye", "watering left eye",
        "watery right eye", "watery left eye", "excessive tearing", "eye watering", "tearing of eye", "eyes watering constantly"
    ],
    "eye redness": [
        "eye redness", "red eye", "red right eye", "red left eye",
        "bloodshot eye", "eyes red", "redness of eye", "pink eye"
    ],
    "itchiness of eye": [
        "itchiness of eye", "itchy eye", "itchy right eye", "itchy left eye",
        "eye itching", "itching in eye", "itchy eyes"
    ],
    "foot or toe swelling": [
        "foot or toe swelling", "swollen big toe", "swelling in big toe",
        "swollen foot", "toe swelling", "swollen toe", "swollen joint in toe", "big toe is swollen"
    ],
    "foot or toe pain": [
        "foot or toe pain", "pain in big toe", "big toe pain",
        "toe pain", "foot pain", "sudden severe pain in big toe",
        "painful big toe", "pain in toe", "severe pain in right big toe", "severe pain in big toe"
    ],
    # Peripheral neuropathy — tingling must map to paresthesia, NOT loss of sensation
    "paresthesia": [
        "paresthesia", "tingling", "tingling in feet", "tingling in hands",
        "tingling in fingers", "tingling in toes", "pins and needles",
        "pins and needles in feet", "numbness and tingling", "prickling sensation",
        "burning tingling sensation", "tingling sensation in limbs"
    ],
    "skin rash": [
        "skin rash", "rash", "itchy red spots", "red spots", "blisters all over", "fluid-filled blisters",
        "little blisters", "itchy rash", "skin eruption", "spots on skin", "red bumps"
    ],
    "itching of skin": [
        "itching of skin", "itchy skin", "itchiness", "pruritus", "severe itching", "scratching skin"
    ],
    "fatigue": [
        "fatigue", "tiredness", "exhaustion", "feeling tired", "super tired", "lack of energy", "lethargy"
    ],
    "chills": [
        "chills", "feeling cold", "shivering", "feeling chilly", "always feeling cold"
    ],
    "weight gain": [
        "weight gain", "gained weight", "gaining weight", "unexplained weight gain", "weight increase"
    ],
}

FEATURE_SYNONYMS: Dict[str, List[str]] = {}
try:
    load_path = SYNONYMS_FILE if SYNONYMS_FILE.exists() else FALLBACK_SYNONYMS_FILE
    if load_path.exists():
        with open(load_path, "r", encoding="utf-8") as f:
            FEATURE_SYNONYMS = json.load(f)
        LOGGER.info("Loaded base synonyms from %s (%d entries)", load_path, len(FEATURE_SYNONYMS))
    for k, v in CURATED_SYNONYMS.items():
        clean_k = k.lower().replace("_", " ")
        if clean_k in FEATURE_SYNONYMS:
            FEATURE_SYNONYMS[clean_k] = list(dict.fromkeys(FEATURE_SYNONYMS[clean_k] + v))
        else:
            FEATURE_SYNONYMS[clean_k] = v
except Exception as e:
    LOGGER.warning(f"Failed to load synonyms: {e}")
    FEATURE_SYNONYMS = CURATED_SYNONYMS


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question about the uploaded healthcare documents")


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


class DiagnoseResponse(BaseModel):
    predicted: str
    probability: float
    probabilities: Optional[Dict[str, float]] = None


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


state = AppState()


def build_prompt(query: str, context_chunks: List[str]) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    return (
        "You are an expert clinical AI. Using the retrieved context, provide a highly detailed, "
        "comprehensive, and multi-paragraph answer to the user's question. Explain the reasoning clearly.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Detailed Answer:"
    )


def map_extracted_to_features(extracted_list: List[str], valid_features: List[str]) -> List[str]:
    final_symptoms = set()
    unmapped_symptoms = []
    feature_lower = [f.lower().replace("_", " ") for f in valid_features]

    for symptom in extracted_list:
        symptom_clean = str(symptom).lower().strip()
        if len(symptom_clean) < 4:
            unmapped_symptoms.append(symptom_clean)
            continue
            
        LOGGER.info("Mapping extracted symptom: '%s'", symptom_clean)

        # 1. Fast-path: Exact or near-exact difflib string match
        best_ratio = 0
        best_orig = None
        for orig, lowered in zip(valid_features, feature_lower):
            ratio = difflib.SequenceMatcher(None, symptom_clean, lowered).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_orig = orig

        if best_ratio > 0.9 and best_orig:
            LOGGER.info("  -> Selected via fast-path (ratio=%.3f): '%s'", best_ratio, best_orig)
            final_symptoms.add(best_orig)
            continue
            
        # 2. Semantic Embedding Similarity Match
        if state.embeddings is None or state.feature_embeddings is None or state.embedding_to_feature is None:
            LOGGER.warning("  -> Dropped (Embeddings not initialized)")
            continue
            
        phrase_emb = np.array(state.embeddings.embed_query(symptom_clean)).reshape(1, -1)
        sims = cosine_similarity(phrase_emb, state.feature_embeddings)[0]
        
        top_indices = sims.argsort()[-5:][::-1]
        
        for idx in top_indices:
            LOGGER.info("  Candidate: '%s' (score: %.3f)", state.embedding_to_feature[idx], sims[idx])

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
            # Eye context -> Genital feature
            if phrase_has_eye and any(g in candidate_lower for g in GENITAL_FEATURES): return True
            # Nasal context -> Eye feature or Genital feature
            if phrase_has_nasal and any(g in candidate_lower for g in EYE_FEATURES.union(GENITAL_FEATURES)): return True
            # Taste context -> Mouth pain feature
            if phrase_has_taste and any(g in candidate_lower for g in MOUTH_PAIN_FEATURES): return True
            return False

        valid_candidates = []
        seen_features = set()
        
        for idx in top_indices:
            candidate_orig = state.embedding_to_feature[idx]
            candidate_lower = candidate_orig.lower()
            score = sims[idx]
            
            if score < 0.65:
                continue
                
            if is_conflict(symptom_clean, candidate_lower):
                LOGGER.warning("  -> Rejected '%s' for phrase '%s' (anatomical/context conflict).", candidate_orig, symptom_clean)
                continue
                
            if candidate_orig not in seen_features:
                valid_candidates.append((candidate_orig, score))
                seen_features.add(candidate_orig)

        if not valid_candidates:
            LOGGER.info("  -> Unmapped (no valid candidates >= 0.65 or all conflicted)")
            unmapped_symptoms.append(symptom_clean)
            continue
            
        best_orig, best_score = valid_candidates[0]
        
        if len(valid_candidates) > 1:
            runner_up_orig, runner_up_score = valid_candidates[1]
            if (best_score - runner_up_score) <= 0.04:
                # Margin near-tie
                LOGGER.warning("  -> Rejected '%s' due to margin near-tie with '%s' (diff: %.3f)", best_orig, runner_up_orig, best_score - runner_up_score)
                unmapped_symptoms.append(symptom_clean)
                continue

        LOGGER.info("  -> Selected via semantics: '%s' (score: %.3f)", best_orig, best_score)
        final_symptoms.add(best_orig)

    if unmapped_symptoms:
        LOGGER.warning("Symptoms described but not supported by current symptom schema: %s", ", ".join(unmapped_symptoms))

    return list(final_symptoms)

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

def extract_symptoms_with_llm(source_text: str, feature_columns: List[str]) -> List[str]:
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
                "temperature": 0.0,  # Deterministic — eliminates sampling variance across runs
                "num_predict": 512,  # Enough tokens for a full symptom list without truncation
            },
        )

        content_str = response["message"]["content"].strip()
        if "```" in content_str:
            content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
            content_str = re.sub(r"\s*```$", "", content_str)

        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError:
            # Fallback: regex search for JSON object or array
            match = re.search(r"\{.*\}", content_str, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = {}

        if isinstance(parsed, dict):
            extracted_list = parsed.get("symptoms", [])
        elif isinstance(parsed, list):
            extracted_list = parsed
        else:
            extracted_list = []

        return map_extracted_to_features(extracted_list, feature_columns)

    except Exception as e:
        LOGGER.exception("Extraction error: %s", e)
        return []


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

    state.embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )
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

    yield

    state.vector_store = None
    state.embeddings = None
    state.ollama_client = None
    state.text_splitter = None
    state.classifier = None
    state.classifier_features = None


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

    # Build feature vector in the trained order
    feature_names = state.classifier_features
    selected = set([s.lower() for s in req.symptoms or []])

    x = [1 if fname.lower() in selected else 0 for fname in feature_names]

    try:
        import pandas as pd

        input_df = pd.DataFrame([x], columns=feature_names)
        if hasattr(state.classifier, "predict_proba"):
            proba = state.classifier.predict_proba(input_df)[0]
            classes = list(state.classifier.classes_)
            sorted_predictions = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)

            top_pred_class, top_pred_prob = sorted_predictions[0]
            pred = str(top_pred_class)
            probability = float(top_pred_prob)

            # Top 3 predicted diseases and their exact probability percentages
            top_3 = sorted_predictions[:3]
            probs = {str(c): float(p) for c, p in top_3}
        else:
            pred = state.classifier.predict(arr)[0]
            probability = 1.0
            probs = {str(pred): 1.0}

    except Exception as exc:
        LOGGER.exception("Diagnosis failed")
        raise HTTPException(status_code=500, detail="Failed to run diagnosis") from exc

    # Abstain when top confidence is below threshold — surface top candidate
    # in probabilities dict so the frontend can optionally show "X suspected"
    ABSTENTION_THRESHOLD = 0.65
    if probability < ABSTENTION_THRESHOLD:
        LOGGER.info(
            "Abstaining: top prediction '%s' at %.1f%% is below %.0f%% threshold",
            pred, probability * 100, ABSTENTION_THRESHOLD * 100
        )
        pred = "Non-specific symptoms (low confidence)"

    return DiagnoseResponse(predicted=str(pred), probability=probability, probabilities=probs)


@app.post("/api/extract-symptoms")
async def extract_symptoms(request: dict):
    text = request.get("text", "") or request.get("description", "")
    feature_columns = state.classifier_features or []
    if not text:
        return {"symptoms": []}

    matched_symptoms = extract_symptoms_with_llm(text, feature_columns)
    return {"symptoms": matched_symptoms}


@app.post("/api/extract-from-docs")
def extract_symptoms_from_docs() -> Dict[str, List[str]]:
    if state.classifier_features is None:
        raise HTTPException(status_code=404, detail="No symptom feature list available")
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready")

    query_text = "patient symptoms, chief complaint, physical signs, clinical presentation"
    try:
        retrieved_documents = state.vector_store.similarity_search(query_text, k=5)
    except Exception as exc:
        LOGGER.exception("Vector search failed for extract-from-docs")
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


@app.post("/api/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    if state.vector_store is None or state.ollama_client is None:
        raise HTTPException(status_code=503, detail="Backend resources are not ready")

    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        retrieved_documents = state.vector_store.similarity_search(query_text, k=TOP_K)
    except Exception as exc:
        LOGGER.exception("Vector search failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve supporting context") from exc

    if not retrieved_documents:
        raise HTTPException(status_code=404, detail="No relevant context found in the local documents")

    context_chunks = [document.page_content for document in retrieved_documents]
    prompt = build_prompt(query_text, context_chunks)

    try:
        ollama_response = state.ollama_client.chat(
            model=DEFAULT_OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical AI. Using the retrieved context, provide a highly "
                        "detailed, comprehensive, and multi-paragraph answer to the user's question. "
                        "Explain the reasoning clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
        )
    except Exception as exc:
        LOGGER.exception("Ollama generation failed")
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
