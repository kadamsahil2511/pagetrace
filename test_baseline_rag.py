import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from answer import generate_answer


class FakeEmbedder:
    def __init__(self):
        self.passage_calls = 0

    def encode(self, texts, *, prompt_name, normalize_embeddings):
        assert normalize_embeddings
        if prompt_name == "query":
            return np.array([[1.0, 0.0]])
        self.passage_calls += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]])


class FakeReranker:
    def predict(self, pairs):
        return np.array([{"near": 0.1, "far": 0.9, "other": 0.2}[text] for _, text in pairs])


class BaselineRagTests(unittest.TestCase):
    def test_shared_answer_keeps_chunk_citations(self):
        hits = [(0.9, {"text": "The answer is 42.", "page": 2})]
        response = io.BytesIO(b'{"message": {"content": "It is 42 [Chunk 1]."}}')
        with patch("answer.urlopen", return_value=response) as request:
            self.assertEqual(generate_answer("What is it?", hits), "It is 42 [Chunk 1].")
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("[Chunk 1] (page 2) The answer is 42.", payload["messages"][1]["content"])

    def test_index_reuses_complete_collection_and_recovers_partial_index(self):
        from baseline_rag import index_chunks
        import chromadb

        chunks = [
            {"chunk_id": "id1", "text": "near", "page": 1, "source": "/doc.pdf"},
            {"chunk_id": "id2", "text": "far", "page": 2, "source": "/doc.pdf"},
        ]
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            client = chromadb.PersistentClient(path=directory)
            collection = client.get_or_create_collection("test_pdf", embedding_function=None)
            index_chunks(collection, chunks, embedder)
            collection = chromadb.PersistentClient(path=directory).get_collection("test_pdf", embedding_function=None)
            index_chunks(collection, chunks, embedder)
            self.assertEqual(collection.count(), 2)
            self.assertEqual(embedder.passage_calls, 1)
            self.assertEqual(collection.get(ids=["id2"])["metadatas"][0], {"page": 2, "source": "/doc.pdf"})
            collection.delete(ids=["id2"])
            index_chunks(collection, chunks, embedder)
            self.assertEqual(collection.count(), 2)

    def test_retrieve_reranks_candidates_and_preserves_citations(self):
        from baseline_rag import retrieve
        import chromadb

        with tempfile.TemporaryDirectory() as directory:
            client = chromadb.PersistentClient(path=directory)
            collection = client.get_or_create_collection("test_pdf", embedding_function=None)
            collection.add(
                ids=["id1", "id2", "id3"],
                documents=["near", "far", "other"],
                embeddings=[[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
                metadatas=[
                    {"page": 1, "source": "/doc.pdf"},
                    {"page": 2, "source": "/doc.pdf"},
                    {"page": 3, "source": "/doc.pdf"},
                ],
            )
            hits = retrieve("question", collection, FakeEmbedder(), FakeReranker(), candidate_k=3, k=2)
            self.assertEqual([chunk["chunk_id"] for _, chunk in hits], ["id2", "id3"])
            self.assertEqual([chunk["page"] for _, chunk in hits], [2, 3])
            self.assertAlmostEqual(hits[0][0], 0.9)

    def test_missing_pdf_exits_with_readable_error(self):
        from baseline_rag import main

        with patch("sys.argv", ["baseline_rag.py", "/no/such/document.pdf"]):
            with self.assertRaisesRegex(SystemExit, "Cannot read PDF"):
                main()


if __name__ == "__main__":
    unittest.main()
