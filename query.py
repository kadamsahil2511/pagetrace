"""Search the PDF chunks saved by loader.py."""

import pickle
import sys
import warnings
from textwrap import shorten, wrap

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from loader import CHUNKS_PATH, INDEX_PATH, MODEL


def retrieve(question, index, chunks, model, k=4):
    q_vec = model.encode([question], prompt_name="query")
    q_vec = np.asarray(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, min(k, index.ntotal))
    return [(float(score), chunks[idx]) for score, idx in zip(scores[0], idxs[0])]


def main(query: str):
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as source:
        chunks = pickle.load(source)
    print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks.")
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks do not match; run loader.py again")

    warnings.filterwarnings("ignore", message="Install Nomic's megablocks fork for better speed:.*")
    torch.set_num_threads(1)
    model = SentenceTransformer(MODEL, trust_remote_code=True, device="cpu")
    hits = retrieve(query, index, chunks, model)
    print(f"\nQuery: {query}")
    print(f"Top {len(hits)} matches (cosine similarity)\n")
    header = f"| {'Rank':<4} | {'Score':>6} | {'Page':>4} | {'Chunk':<5} | {'Text':<64} |"
    print(header)
    print("-" * len(header))
    for rank, (score, chunk) in enumerate(hits, start=1):
        preview = shorten(chunk["text"], width=180, placeholder="...")
        lines = wrap(preview, width=64)
        short_id = chunk["chunk_id"].split(":")[-1]
        for line_number, line in enumerate(lines):
            first = line_number == 0
            score_text = f"{score:.4f}" if first else ""
            print(f"| {rank if first else '':<4} | {score_text:>6} | {chunk['page'] if first else '':>4} | {short_id if first else '':<5} | {line:<64} |")
        print("-" * len(header))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "What is the main topic of the document?")
