# Context-Aware Retrieval-Augmented Generation for Intelligent Healthcare Document Analysis

A fully local medical document question-answering system that ingests unstructured healthcare files, stores embeddings in a local vector database, and answers questions through a local Ollama model. The design keeps medical data on-device and avoids external cloud APIs.

## Overview

The project has two parts:

- `backend/`: FastAPI service for ingestion, retrieval, local LLM generation, symptom extraction, and disease prediction.
- `frontend/`: Next.js App Router UI with Tailwind CSS for querying documents and reviewing evidence.

## Core Features & Architecture

### 1. Local Document Q&A (RAG)
1. Ingest local `.txt`, `.md`, and `.pdf` medical files.
2. Chunk text with context-aware, boundary-safe splitting.
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store vectors locally in ChromaDB.
5. Retrieve the top 3 chunks for each user query.
6. Send retrieved context to a local Ollama model for answer generation.
7. Return both the answer and source chunks to the UI.

### 2. Intelligent Disease Prediction
1. **Symptom Extraction:** Uses a local LLM to exhaustively extract physical complaints and symptoms from free-text clinical notes or patient descriptions.
2. **Semantic Ontology Mapping:** Extracted free-text symptoms are mapped to a strict 230-feature clinical schema using semantic embeddings (Cosine Similarity) with a fast-path Exact Match (difflib) and curated synonyms. 
3. **Safety Guardrails:** Uses margin-based rejection (dropping near-ties) and anatomical context guards (preventing eye/nasal/mouth crossover hallucinations) to ensure high-fidelity mapping.
4. **Diagnosis Classifier:** A Stacking Ensemble (Random Forests, Gradient Boosting, MLP) trained on 195,000 samples predicts the top diseases based on the activated symptom vector.

### 3. Evidence-Adaptive Clinical Decision Gate (Single Major Innovation Feature)

The **Evidence-Adaptive Clinical Decision Gate (EACDG)** is an integrated decision-gating layer added to the healthcare document-analysis pipeline. It evaluates whether a model-generated clinical prediction is sufficiently supported by the extracted and retrieved clinical evidence, rather than relying solely on machine learning model confidence.

> **Core Principle:** *Model confidence is not necessarily the same as evidence sufficiency.*

#### Position in the Existing Architecture

| Stage | Function | Status |
|---|---|---|
| **Medical Document** | Input healthcare PDF / clinical document | Existing |
| **Document Parsing** | Converts document into usable text (`PyPDF` / boundary-safe splitter) | Existing |
| **LLM Symptom Extraction** | Exhaustively identifies physical symptoms and complaints from text | Existing |
| **Semantic Symptom Mapping** | Maps extracted expressions to 230-feature clinical schema | Existing |
| **ML Classifier** | Produces disease/class prediction and softmax confidence | Existing |
| **Evidence-Adaptive Clinical Decision Gate** | **Evaluates evidence sufficiency before accepting the prediction** | **NEW SINGLE FEATURE** |
| **RAG** | Retrieves relevant evidence chunks and supports the final response | Existing |

#### Complete Data Flow

```text
Medical Document
       ↓
Document Parsing / Text Extraction
       ↓
LLM Symptom Extraction
       ↓
Semantic Symptom Mapping
       ↓
Structured Symptom Representation (230-dim vector x)
       ↓
Existing ML Classifier (Predictions & Softmax Probabilities)
       ↓
Evidence-Adaptive Clinical Decision Gate (EACDG)
  ├── Sufficient Evidence   → [DIAGNOSE] Accept Prediction
  ├── Borderline / Sparse   → [CLARIFY] Targeted Information-Gain Query
  └── Insufficient Evidence → [ABSTAIN] Flag & Output Clinical Rationale
       ↓
RAG-supported Final Response
```

#### Core Components & Mathematical Formulations

1. **Three-State Epistemic Symptom Classification:**
   - **`SUPPORTED`**: High-confidence match ($\text{Cosine} \ge 0.70$ or $\text{Difflib} > 0.90$, margin $\Delta > 0.04$) entering the diagnostic feature vector $x$.
   - **`UNCERTAIN`**: Borderline or near-tie match ($0.58 \le \text{Cosine} < 0.70$ or $\Delta \le 0.04$) retained for evidence gating and targeted clarification.
   - **`UNSUPPORTED`**: Described by patient but absent from the 230-feature schema ($< 0.58$ or anatomical conflict). Logged as honest schema gaps (e.g., photophobia, sensory aura) rather than silently forced into false matches.

2. **Diagnostic Evidence Coverage Engine ($EC$):**
   Measures how much of the evidence considered relevant to the predicted disease is supported by the patient's extracted symptoms:
   $$\text{Evidence Coverage } (EC) = \left( \frac{\sum_{j \in \mathcal{H}_d} w_{d, j} \cdot \mathbb{I}(x_j = 1)}{\sum_{j \in \mathcal{H}_d} w_{d, j}} \right) \times 100$$
   - Empirical conditional symptom frequencies $f_{d, j} = P(\text{symptom}_j = 1 \mid \text{disease} = d)$ are profiled across all 100 diseases from `Diseases_and_Symptoms_dataset.csv`.
   - Hallmark features are indexed where $f_{d, j} \ge 0.40$ with hallmark weights $w_{d, j} = f_{d, j}$.

3. **Distributional Sparsity Mismatch Metric ($S$ & $z$-score):**
   - Live vector sparsity: $S = 1 - \frac{\|x\|_0}{230}$.
   - Domain shift $z$-score relative to the SMOTE training distribution ($\mu = 5.87, \sigma = 1.67$):
     $$z_{\text{sparsity}} = \frac{\|x\|_0 - 5.87}{1.67}$$
   - Quantifies when sparse clinical presentations (1–3 complaints) are at risk of synthetic training density mismatch.

4. **Adaptive Multi-Criteria Decision Gate ($T = f(S, A, C, N)$):**
   Decouples statistical classifier confidence from clinical evidence coverage and dynamically transitions across three actionable states:
   - `DIAGNOSE`: High confidence, evidence coverage $\ge 35\%$, and at least 1 hallmark symptom present.
   - `CLARIFY`: Moderate coverage ($15\% \le EC < 35\%$) or sparse domain mismatch where a follow-up query reduces uncertainty.
   - `ABSTAIN`: Sub-floor confidence ($< 65\%$) or severe evidence deficit ($EC < 15\%$), returning an explicit auditable clinical rationale.

5. **Discriminative Information-Gain Clarification Engine:**
   - Evaluates hallmark divergence $|f_{d_1, j} - f_{d_2, j}|$ between top competing hypotheses $(d_1, d_2)$.
   - Selects the single most discriminative missing hallmark feature and prompts the user with one-click `✓ Yes` / `✕ No` vector updating.

6. **Self-Audited Clinical Synonym Pipeline (`audit_synonyms.py`):**
   - Automated audit verifying candidate terms against semantic drift, anatomical conflict, and feature collisions.
   - Outputs verified, versioned `approved_synonyms_v1.json` (467 approved terms, SHA-256: `800cc1914561`).

#### Implementation Boundary

- **Module Boundary:** The EACDG is an integrated decision layer (`backend/evidence_engine.py`) downstream of symptom extraction and ML classification. Existing extraction components remain responsible for producing symptoms; the gate consumes their outputs without re-parsing the document.
- **Gate Inputs:** Predicted disease/class, classifier confidence, mapped patient symptoms (supported, uncertain, unsupported), disease hallmark profiles, and sparsity metrics.
- **Gate Outputs:** Accepted prediction (`DIAGNOSE`), targeted clarification question (`CLARIFY`), or abstention decision (`ABSTAIN`) with full clinical evidence breakdown.

#### Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Accuracy / Precision / Recall / F1** | Check predictive classification performance |
| **Calibration Error / Brier Score** | Check probability confidence reliability |
| **Evidence Coverage ($EC$)** | Quantify supporting clinical hallmark evidence available |
| **Abstention Rate** | Measure rejection of insufficiently supported predictions |
| **False-Positive Diagnosis Rate** | Measure reduction of overconfident, unsupported predictions |
| **Uncertain / Unsupported Mapping Rate** | Measure semantic extraction & mapping quality |

#### Project & Viva Positioning

- **Main Project:** Context-Aware Retrieval-Augmented Generation Model for Intelligent Healthcare Document Analysis
- **Single Major Innovation Feature:** Evidence-Adaptive Clinical Decision Gate (EACDG)
- **Viva Explanation:** *"The proposed contribution is an evidence-adaptive decision-gating mechanism that determines whether a model-generated clinical prediction is sufficiently supported by the extracted and retrieved evidence, rather than relying solely on model confidence."*


## Known Problems & Solutions

### Problem 1: PDF parsing and extracting information from documents is not working properly
Standard PDF extraction tools lose tabular structure, header context, and spacing, causing the LLM to hallucinate or misinterpret medical histories.

**Solution:** 
We utilize libraries like `PyPDF` / `pdfplumber` for strict spatial extraction, combined with a context-aware recursive chunking strategy. The text is broken at natural semantic boundaries (paragraphs, bullet points) rather than arbitrary character limits to ensure medical context remains intact when embedded into ChromaDB.

### Problem 2: Disease prediction from symptoms is not working as expected
Initially, asking the LLM to directly map symptoms to the dataset schema caused severe hallucinations, over-matching, and misclassification (e.g., predicting diseases based on fabricated symptoms). Furthermore, the dataset contains sparse vectors and lacks certain critical columns.

**Solution:**
We separated extraction from mapping. The LLM is strictly prompted to extract the *exact* physical symptoms from the text. Those raw symptoms are then deterministically mapped to the dataset's exact 230 columns using semantic similarity scoring, a curated synonym dictionary, and margin-based rejection to drop ambiguous matches.

**Extraction Prompt Used:**
```text
You are a thorough medical symptom extractor. Your task is to read the text below and extract EVERY SINGLE symptom or physical complaint mentioned, without exception.

Rules:
1. Extract ALL symptoms — do not stop after the first one. Read the entire text and list every symptom.
2. Preserve anatomical and contextual specificity: write "swelling in big toe" not "swelling", "crusty eye discharge" not "discharge", "gritty eye sensation" not "eye discomfort".
3. Treat each distinct complaint as a separate item (e.g. redness, itchiness, discharge, gritty sensation, watering are all separate entries).
4. Do NOT summarize or group symptoms together.
5. Respond ONLY with a valid JSON object: {{"symptoms": ["symptom 1", "symptom 2", ...]}}

Example for a text describing 5 symptoms: {{"symptoms": ["runny nose", "congestion", "scratchy throat", "coughing", "sneezing"]}}

Text: {source_text}
```

## Backend Setup

### Requirements

- Python 3.10 or newer
- Ollama installed locally
- A pulled local model, such as `llama3.1` or `mistral`

### Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Add documents

Place local medical source files in:

```text
backend/data/raw/
```

Supported formats:

- `.txt`
- `.md`
- `.pdf`

### Run ingestion

```bash
cd backend
python ingest.py --reset-index
```

This creates or refreshes the local Chroma index in `backend/chroma_db/`.

### Start the FastAPI server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

### Install dependencies

```bash
cd frontend
npm install
```

### Start the UI

```bash
cd frontend
npm run dev
```

The UI runs on `http://localhost:3000`.

## Environment Notes

The backend uses these local defaults:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Chroma collection: `medical_documents`
- Ollama model: `llama3.1` by default

You can change the Ollama model with:

```bash
export OLLAMA_MODEL=mistral
```

## Testing & Verification

The repository includes a complete automated test and clinical benchmark suite:

### 1. Evidence Engine Unit Tests
Runs 7 targeted unit tests covering hallmark extraction, sparsity $z$-score computation, evidence coverage calculations, and decision gate logic:

```bash
python3 backend/test_evidence_engine.py
```

### 2. End-to-End Clinical Benchmarks
Evaluates 4 real-world clinical presentations against the evidence-adaptive decision gate:

```bash
python3 backend/test_pipeline_e2e.py
```

| Benchmark Case | Model Softmax | Evidence Coverage | Decision Output | Verification Goal |
|---|---|---|---|---|
| **Conjunctivitis (4 hallmarks)** | 99.0% | **35.4%** | `[DIAGNOSE]` | High coverage allows definitive diagnosis |
| **Migraine Trap** (`headache, nausea`) | 91.0% | **17.9%** | `[CLARIFY]` | Prevents 91% false Hyperemesis Gravidarum diagnosis |
| **Diabetic Trap** (`frequent urination`) | 88.0% | **8.6%** | `[CLARIFY]` | Detects missing BPH hallmarks |
| **Gout (Sparse toe pain)** | 53.0% | **17.1%** | `[ABSTAIN]` | Honest abstention on low confidence |

### 3. Clinical Synonym Audit Pipeline
Verifies candidate lay terms against semantic drift and anatomical collisions, exporting `approved_synonyms_v1.json`:

```bash
python3 backend/audit_synonyms.py
```

### 4. Frontend Typecheck & Production Build
Ensures clean TypeScript compilation and Next.js static prerendering:

```bash
cd frontend
npx tsc --noEmit
npm run build
```

## Design Notes

- **Clinical Visual Hierarchy:** White and deep plum clinical palette with semantic color coding: emerald for `SUPPORTED` symptoms / `DIAGNOSE`, amber for `UNCERTAIN` symptoms / `CLARIFY`, and slate for `UNSUPPORTED` / `ABSTAIN`.
- **Dual Gauge Inspection:** Displays model classifier confidence alongside Diagnostic Evidence Coverage to visually communicate why high confidence alone is gated when clinical hallmark coverage is deficient.
- **Interactive Information-Gain Loop:** When the decision gate flags `CLARIFY`, the UI renders the single most discriminative question with quick `✓ Yes` / `✕ No` action buttons that update the feature vector and re-run diagnosis automatically.
- **Privacy & Local Execution:** The system is strictly on-device, storing embeddings in local ChromaDB and using local Ollama LLMs to maintain patient data sovereignty.

## Things to be Done / Improved (Future Work)

- **Dataset Enrichment:** Source a richer dataset that natively includes critical missing features (e.g., weight loss, sensory auras, polydipsia) and naturally contains realistic variance in symptom sparsity.
- **Clinical Ontology Layer:** Implement an ICD-10 or SNOMED-CT mapping layer to replace pure cosine-similarity, completely eliminating near-tie ambiguity.
- **Minimum Margin Requirement:** Continue refining the margin-based rejection thresholds to flag ambiguous cases for human review rather than silently guessing.
- **UI Enhancements:** Add a document upload form, query history, and citation highlighting in the evidence panel.
- **Dockerization:** Add Dockerfiles for fully local, one-click deployment.
