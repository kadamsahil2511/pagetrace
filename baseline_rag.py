"""Chat with one PDF using Chroma retrieval and cross-encoder reranking."""

import hashlib
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
import torch
from pypdf.errors import PdfReadError
from sentence_transformers import CrossEncoder, SentenceTransformer

from answer import OLLAMA_MODEL, generate_answer
from loader import MODEL, text_split_pdf_by_chunk_size


DB_PATH = Path(__file__).with_name(".chroma")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def index_chunks(collection, chunks, embedder):
    if collection.count() == len(chunks):
        return
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        embeddings = embedder.encode(
            [chunk["text"] for chunk in batch], prompt_name="passage", normalize_embeddings=True
        )
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[{"page": chunk["page"], "source": chunk["source"]} for chunk in batch],
            embeddings=embeddings.tolist(),
        )


def retrieve(question, collection, embedder, reranker, candidate_k=20, k=4):
    count = collection.count()
    if not count:
        return []
    query_embedding = embedder.encode([question], prompt_name="query", normalize_embeddings=True)
    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(candidate_k, count),
        include=["documents", "metadatas"],
    )
    candidates = [
        {"chunk_id": chunk_id, "text": text, **metadata}
        for chunk_id, text, metadata in zip(result["ids"][0], result["documents"][0], result["metadatas"][0])
    ]
    if not candidates:
        return []
    scores = reranker.predict([(question, chunk["text"]) for chunk in candidates])
    return sorted(
        [(float(score), chunk) for score, chunk in zip(scores, candidates)],
        key=lambda hit: hit[0], reverse=True,
    )[:k]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python baseline_rag.py path/to/document.pdf")

    try:
        chunks = text_split_pdf_by_chunk_size(sys.argv[1], CHUNK_SIZE, CHUNK_OVERLAP)
    except (OSError, PdfReadError) as error:
        raise SystemExit(f"Cannot read PDF: {error}") from error
    if not chunks:
        raise SystemExit("No extractable text found in the PDF")
    pdf_hash = chunks[0]["chunk_id"].split(":", 1)[0]
    identity = f"{chunks[0]['source']}|{pdf_hash}|{MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"
    collection_name = "pdf_" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(collection_name, embedding_function=None)

    warnings.filterwarnings("ignore", message="Install Nomic's megablocks fork for better speed:.*")
    torch.set_num_threads(1)
    embedder = SentenceTransformer(MODEL, trust_remote_code=True, device="cpu")
    index_chunks(collection, chunks, embedder)
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    print(f"Chroma RAG chatbot ready: {collection.count()} chunks, Ollama model {OLLAMA_MODEL}.")
    print("Type 'quit' to exit. Each question is answered from reranked PDF chunks.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        hits = retrieve(question, collection, embedder, reranker)
        if not hits:
            print("Chatbot: I don't know.\n")
            continue
        print(f"Chatbot: {generate_answer(question, hits)}\n")
        print("Sources:")
        for number, (score, chunk) in enumerate(hits, start=1):
            short_id = chunk["chunk_id"].split(":")[-1]
            print(f"  [Chunk {number}] page={chunk['page']} chunk={short_id} rerank_score={score:.3f}")
        print()


if __name__ == "__main__":
    main()
