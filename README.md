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

## Design Notes

- The UI uses a clean white and deep purple clinical theme.
- Evidence is shown alongside the answer so users can verify the retrieved source text.
- The system is intended to stay fully local for privacy-sensitive healthcare workflows.

## Things to be Done / Improved (Future Work)

- **Dataset Enrichment:** Source a richer dataset that natively includes critical missing features (e.g., weight loss, sensory auras, polydipsia) and naturally contains realistic variance in symptom sparsity.
- **Clinical Ontology Layer:** Implement an ICD-10 or SNOMED-CT mapping layer to replace pure cosine-similarity, completely eliminating near-tie ambiguity.
- **Minimum Margin Requirement:** Continue refining the margin-based rejection thresholds to flag ambiguous cases for human review rather than silently guessing.
- **UI Enhancements:** Add a document upload form, query history, and citation highlighting in the evidence panel.
- **Dockerization:** Add Dockerfiles for fully local, one-click deployment.
