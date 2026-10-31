"""Offline PDF chatbot: FAISS retrieval followed by local Ollama generation."""

import os
import pickle
import warnings

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import torch
from sentence_transformers import SentenceTransformer

from answer import OLLAMA_MODEL, generate_answer
from loader import CHUNKS_PATH, INDEX_PATH, MODEL
from query import retrieve


def main():
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as source:
        chunks = pickle.load(source)
    if index.ntotal != len(chunks):
        raise SystemExit("Index and chunks do not match; run loader.py again")

    warnings.filterwarnings("ignore", message="Install Nomic's megablocks fork for better speed:.*")
    torch.set_num_threads(1)
    model = SentenceTransformer(MODEL, trust_remote_code=True, device="cpu")
    print(f"Offline RAG chatbot ready: {len(chunks)} chunks, Ollama model {OLLAMA_MODEL}.")
    print("Type 'quit' to exit. Each question is answered from retrieved PDF chunks.\n")

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

        hits = retrieve(question, index, chunks, model)
        cover = chunks[0]
        if all(chunk["chunk_id"] != cover["chunk_id"] for _, chunk in hits):
            hits.insert(0, (None, cover))
        print(f"Chatbot: {generate_answer(question, hits)}\n")
        print("Sources:")
        for number, (score, chunk) in enumerate(hits, start=1):
            short_id = chunk["chunk_id"].split(":")[-1]
            score_text = f"{score:.3f}" if score is not None else "cover"
            print(f"  [Chunk {number}] page={chunk['page']} chunk={short_id} score={score_text}")
        print()


if __name__ == "__main__":
    main()
