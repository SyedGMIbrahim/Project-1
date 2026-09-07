from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pandas as pd

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "medical_documents"
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "mentor_dataset"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(file_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))
    page_texts: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append((page_number, text))

    return page_texts


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        keep_separator=True,
        is_separator_regex=False,
    )


def load_documents(input_dir: Path) -> list[Document]:
    source_documents: list[Document] = []

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            page_texts = read_pdf_file(file_path)
            for page_number, page_text in page_texts:
                cleaned_text = page_text.strip()
                if not cleaned_text:
                    continue

                source_documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": "pdf",
                            "page": page_number,
                            "page_number": page_number,
                        },
                    )
                )
            continue

        if file_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                LOGGER.exception("Failed to read CSV %s: %s", file_path, exc)
                continue

            # Normalize column names and handle missing columns
            cols = {c.lower(): c for c in df.columns}
            # prefer 'transcription', fallback to 'text'
            text_col = cols.get("transcription") or cols.get("text")
            specialty_col = cols.get("medical_specialty")
            sample_name_col = cols.get("sample_name")
            keywords_col = cols.get("keywords")

            if text_col is None:
                LOGGER.warning("CSV %s does not contain 'transcription' or 'text' column; skipping", file_path)
                continue

            # Replace NaN with empty string
            df = df.fillna("")

            for row_index, row in df.iterrows():
                text = str(row[text_col]).strip()
                if not text:
                    continue

                metadata: dict = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "csv",
                    "row_index": int(row_index),
                }

                # Attach requested metadata fields if present
                if specialty_col:
                    metadata["medical_specialty"] = str(row[specialty_col]).strip()
                else:
                    metadata["medical_specialty"] = ""

                if sample_name_col:
                    metadata["sample_name"] = str(row[sample_name_col]).strip()
                else:
                    metadata["sample_name"] = ""

                if keywords_col:
                    metadata["keywords"] = str(row[keywords_col]).strip()
                else:
                    metadata["keywords"] = ""

                source_documents.append(Document(page_content=text, metadata=metadata))
            continue

        text = read_text_file(file_path).strip()
        if text:
            source_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower().lstrip("."),
                        "page": None,
                        "page_number": None,
                    },
                )
            )

    return source_documents


def ingest_documents(
    input_dir: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    reset_index: bool,
) -> Chroma:
    source_documents = load_documents(input_dir)
    if not source_documents:
        raise ValueError(f"No supported documents found in {input_dir}")

    if reset_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    chunked_documents = splitter.split_documents(source_documents)

    for chunk_index, document in enumerate(chunked_documents, start=1):
        document.metadata["chunk_index"] = chunk_index
        document.metadata["chunk_size"] = len(document.page_content)

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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with clinical documents, such as the mentor dataset.")
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

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    vector_store = ingest_documents(
        input_dir=args.input_dir,
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
