from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "medical_documents"
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_MENTOR_DIR = BASE_DIR / "data" / "mentor_dataset"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Common ligature replacements in OCR / PDF exports
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\xa0": " ",
}


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace, ligatures, and formatting artifacts in extracted text."""
    if not text:
        return ""

    for k, v in LIGATURE_MAP.items():
        text = text.replace(k, v)

    # Normalize carriage returns and line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_table_as_markdown(table: List[List[Optional[str]]]) -> str:
    """Convert a 2D list of table cells into a clean Markdown table."""
    if not table or len(table) < 2:
        return ""

    # Filter out completely empty rows
    cleaned_rows = []
    for row in table:
        cells = [str(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):
            cleaned_rows.append(cells)

    if len(cleaned_rows) < 2:
        return ""

    max_cols = max(len(r) for r in cleaned_rows)
    # Pad rows to uniform column length
    uniform_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    header = uniform_rows[0]
    separator = [":---"] * max_cols
    body = uniform_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def read_pdf_file(file_path: Path) -> List[Tuple[int, str]]:
    """Extract text and structured tables from a PDF using pdfplumber if available,
    with fallback to layout-aware pypdf extraction.
    """
    page_texts: List[Tuple[int, str]] = []

    # Attempt pdfplumber for high-fidelity spatial & tabular extraction
    try:
        import pdfplumber

        with pdfplumber.open(str(file_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_parts: List[str] = []

                # 1. Extract structured tables as Markdown
                tables = page.extract_tables() or []
                for table in tables:
                    md_table = format_table_as_markdown(table)
                    if md_table:
                        page_parts.append(md_table)

                # 2. Extract spatial text
                text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or ""
                cleaned = clean_extracted_text(text)
                if cleaned:
                    page_parts.append(cleaned)

                combined = "\n\n".join(page_parts).strip()
                if combined:
                    page_texts.append((page_number, combined))

        if page_texts:
            return page_texts
    except (ImportError, Exception) as exc:
        LOGGER.debug("pdfplumber extraction skipped or unavailable (%s), using pypdf", exc)

    # Robust fallback: pypdf with layout mode and text cleaning
    try:
        reader = PdfReader(str(file_path))
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                text = page.extract_text() or ""

            cleaned = clean_extracted_text(text)
            if cleaned:
                page_texts.append((page_number, cleaned))
    except Exception as exc:
        LOGGER.error("Failed to read PDF %s: %s", file_path, exc)

    return page_texts


def read_text_file(file_path: Path) -> str:
    """Read a plain text or Markdown file with error tolerance."""
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    return clean_extracted_text(raw)


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build a context-aware recursive splitter with clinical section boundaries."""
    clinical_separators = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\nHISTORY OF PRESENT ILLNESS:",
        "\nPAST MEDICAL HISTORY:",
        "\nREVIEW OF SYSTEMS:",
        "\nPHYSICAL EXAMINATION:",
        "\nLAB PANEL RESULTS:",
        "\nLABORATORY DATA:",
        "\nASSESSMENT & PLAN:",
        "\nIMPRESSION/PLAN:",
        "\nCHIEF COMPLAINT:",
        "\nMEDICATIONS:",
        "\nALLERGIES:",
        "\nDISCHARGE SUMMARY:",
        "\n\n",
        "\n- ",
        "\n* ",
        "\n• ",
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "\n4. ",
        "\n5. ",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        "",
    ]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=clinical_separators,
        keep_separator=True,
        is_separator_regex=False,
    )


def load_documents_from_directory(directory: Path) -> List[Document]:
    """Load documents from a single directory."""
    source_documents: List[Document] = []
    if not directory.exists():
        return source_documents

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            page_texts = read_pdf_file(file_path)
            for page_number, page_text in page_texts:
                cleaned_text = page_text.strip()
                if not cleaned_text:
                    continue

                source_documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_path.name,
                            "file_type": "pdf",
                            "page": page_number,
                            "page_number": page_number,
                        },
                    )
                )
            continue

        if suffix == ".csv":
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                LOGGER.exception("Failed to read CSV %s: %s", file_path, exc)
                continue

            cols = {c.lower(): c for c in df.columns}
            text_col = cols.get("transcription") or cols.get("text")
            specialty_col = cols.get("medical_specialty")
            sample_name_col = cols.get("sample_name")
            keywords_col = cols.get("keywords")

            if text_col is None:
                LOGGER.warning("CSV %s does not contain 'transcription' or 'text' column; skipping", file_path)
                continue

            df = df.fillna("")

            for row_index, row in df.iterrows():
                text = clean_extracted_text(str(row[text_col]))
                if not text:
                    continue

                metadata: dict = {
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_type": "csv",
                    "row_index": int(row_index),
                    "medical_specialty": str(row[specialty_col]).strip() if specialty_col else "",
                    "sample_name": str(row[sample_name_col]).strip() if sample_name_col else "",
                    "keywords": str(row[keywords_col]).strip() if keywords_col else "",
                }

                source_documents.append(Document(page_content=text, metadata=metadata))
            continue

        # .txt or .md
        text = read_text_file(file_path)
        if text:
            source_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "file_type": suffix.lstrip("."),
                        "page": None,
                        "page_number": None,
                    },
                )
            )

    return source_documents


def load_documents(input_paths: Union[Path, List[Path]]) -> List[Document]:
    """Load documents from one or multiple paths/directories."""
    if isinstance(input_paths, Path):
        input_paths = [input_paths]

    all_documents: List[Document] = []

    for path in input_paths:
        if path.is_file():
            # load single file
            parent_docs = load_documents_from_directory(path.parent)
            for doc in parent_docs:
                if doc.metadata.get("file_name") == path.name:
                    all_documents.append(doc)
        elif path.is_dir():
            docs = load_documents_from_directory(path)
            all_documents.extend(docs)

    return all_documents


def ingest_documents(
    input_paths: Union[Path, List[Path]],
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    reset_index: bool,
) -> Chroma:
    """Ingest medical documents into local ChromaDB with embedding and chunking."""
    source_documents = load_documents(input_paths)
    if not source_documents:
        raise ValueError(f"No supported documents found in {input_paths}")

    if reset_index and persist_dir.exists():
        LOGGER.info("Resetting existing Chroma index at %s", persist_dir)
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading embedding model %s...", embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"local_files_only": True},
    )
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    chunked_documents = splitter.split_documents(source_documents)

    for chunk_index, document in enumerate(chunked_documents, start=1):
        document.metadata["chunk_index"] = chunk_index
        document.metadata["chunk_size"] = len(document.page_content)

    LOGGER.info(
        "Split %d source documents into %d context-aware chunks.",
        len(source_documents),
        len(chunked_documents),
    )

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    # Insert documents in batches to avoid exceeding Chroma's max batch size
    batch_size = 1000
    total = len(chunked_documents)
    if total == 0:
        LOGGER.warning("No chunked documents to insert into the vector store")
    else:
        num_batches = (total + batch_size - 1) // batch_size
        LOGGER.info("Adding %d documents to Chroma in %d batches (batch_size=%d)", total, num_batches, batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, total)
            batch = chunked_documents[start:end]
            vector_store.add_documents(batch)
            LOGGER.info("Inserted batch %d/%d (%d documents)", i + 1, num_batches, len(batch))

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local medical documents into a Chroma vector store.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional specific input directory. If not passed, both data/raw and data/mentor_dataset are indexed.",
    )
    parser.add_argument("--persist-dir", type=Path, default=BASE_DIR / "chroma_db", help="Directory used to persist the Chroma index.")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Local Hugging Face embedding model.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Character overlap between chunks.")
    parser.add_argument("--reset-index", action="store_true", help="Delete the existing vector store before ingesting.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.input_dir:
        input_paths = [args.input_dir]
    else:
        input_paths = [DEFAULT_RAW_DIR, DEFAULT_MENTOR_DIR]

    valid_paths = [p for p in input_paths if p.exists()]
    if not valid_paths:
        raise FileNotFoundError(f"No input directories exist among: {input_paths}")

    LOGGER.info("Starting ingestion for paths: %s", [str(p) for p in valid_paths])
    vector_store = ingest_documents(
        input_paths=valid_paths,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset_index=args.reset_index,
    )

    LOGGER.info("Ingestion complete. Collection '%s' is ready at '%s'.", args.collection_name, args.persist_dir)
    LOGGER.info("Vector store type: %s", type(vector_store).__name__)


if __name__ == "__main__":
    main()
