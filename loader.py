"""Load a PDF, split it into chunks, and save embeddings in FAISS."""

import hashlib
import pickle
import sys
from pathlib import Path
from unicodedata import normalize

import faiss
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


MODEL = "nomic-ai/nomic-embed-text-v2-moe"
CHUNKS_PATH = Path(__file__).with_name("chunks.pkl")
INDEX_PATH = Path(__file__).with_name("faiss.index")


def text_split_pdf_by_chunk_size(pdf_file, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Return page-aware chunk records from a PDF."""
    pdf_path = Path(pdf_file).resolve(strict=True)
    reader = PdfReader(pdf_path)
    print(f"Loaded {len(reader.pages)} pages")
    with pdf_path.open("rb") as pdf_file:
        pdf_hash = hashlib.file_digest(pdf_file, "sha256").hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize("NFKC", page.extract_text(extraction_mode="layout") or "")
        text = " ".join(text.split())
        if text:
            for chunk_number, part in enumerate(splitter.split_text(text)):
                chunks.append({
                    "chunk_id": f"{pdf_hash}:p{page_number}:c{chunk_number}",
                    "text": part,
                    "page": page_number,
                    "source": str(pdf_path),
                })
    return chunks


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python loader.py path/to/document.pdf")

    print(f"Loading PDF: {sys.argv[1]}")
    chunks = text_split_pdf_by_chunk_size(sys.argv[1])

    if not chunks:
        raise SystemExit("No extractable text found in the PDF")
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk ID: {chunks[0]['chunk_id']}")

    print(f"Loading model: {MODEL}")
    torch.set_num_threads(1)
    model = SentenceTransformer(MODEL, trust_remote_code=True, device="cpu")
    texts = [chunk["text"] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks ...")
    embeddings = model.encode(texts, prompt_name="passage", show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    print("\n--- What an embedding looks like ---")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First 10 numbers of chunk 0: {embeddings[0][:10]}")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"Built FAISS IndexFlatIP, ntotal = {index.ntotal}")

    faiss.write_index(index, str(INDEX_PATH))
    with CHUNKS_PATH.open("wb") as target:
        pickle.dump(chunks, target)
    print(f"Saved index -> {INDEX_PATH}")
    print(f"Saved chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
