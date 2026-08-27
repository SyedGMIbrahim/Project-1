# Context-Aware Retrieval-Augmented Generation for Intelligent Healthcare Document Analysis

A fully local medical document question-answering system that ingests unstructured healthcare files, stores embeddings in a local vector database, and answers questions through a local Ollama model. The design keeps medical data on-device and avoids external cloud APIs.

## Overview

The project has two parts:

- `backend/`: FastAPI service for ingestion, retrieval, and local LLM generation.
- `frontend/`: Next.js App Router UI with Tailwind CSS for querying documents and reviewing evidence.

## Local Architecture

1. Ingest local `.txt`, `.md`, and `.pdf` medical files.
2. Chunk text with context-aware, boundary-safe splitting.
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store vectors locally in ChromaDB.
5. Retrieve the top 3 chunks for each user query.
6. Send retrieved context to a local Ollama model for answer generation.
7. Return both the answer and source chunks to the UI.

## Project Structure

```text
Project 1/
├── backend/
│   ├── app/
│   ├── chroma_db/
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── ingest.py
│   ├── main.py
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── globals.css
    │   ├── layout.tsx
    │   └── page.tsx
    ├── components/
    ├── public/
    ├── package.json
    ├── tailwind.config.ts
    └── tsconfig.json
```

## Backend Setup

### Requirements

- Python 3.10 or newer
- Ollama installed locally
- A pulled local model, such as `llama3.1` or `mistral`

### Install Python dependencies

```bash
cd "/Users/syedgmibrahim/Desktop/Project 1/backend"
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
cd "/Users/syedgmibrahim/Desktop/Project 1/backend"
python ingest.py --reset-index
```

This creates or refreshes the local Chroma index in:

```text
backend/chroma_db/
```

### Start the FastAPI server

```bash
cd "/Users/syedgmibrahim/Desktop/Project 1/backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Backend endpoint

`POST http://localhost:8000/api/query`

Request body:

```json
{
  "query": "What treatment is recommended for the patient?"
}
```

Response:

```json
{
  "answer": "...",
  "source_documents": [
    {
      "content": "retrieved chunk text",
      "metadata": {
        "source": "path/to/document.pdf",
        "file_name": "document.pdf",
        "file_type": "pdf",
        "page": 2,
        "chunk_index": 1
      }
    }
  ]
}
```

## Frontend Setup

### Install dependencies

```bash
cd "/Users/syedgmibrahim/Desktop/Project 1/frontend"
npm install
```

### Start the UI

```bash
cd "/Users/syedgmibrahim/Desktop/Project 1/frontend"
npm run dev
```

The UI runs on:

```text
http://localhost:3000
```

It sends requests to:

```text
http://localhost:8000/api/query
```

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

## Current Status

Implemented:

- Local document ingestion
- Semantic chunking
- Local ChromaDB storage
- FastAPI query endpoint
- Local Ollama generation
- Next.js query interface
- Evidence verification panel

## Next Steps

Possible follow-ups:

- Add a document upload form
- Add query history
- Add citation highlighting in the evidence panel
- Add Dockerfiles for fully local deployment
