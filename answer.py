"""Generate a PDF-grounded answer with the local Ollama server."""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_MODEL = "gemma4:e2b-mlx"
SYSTEM = (
    "Answer using only the provided context. If the answer is not there, say 'I don't know.' "
    "Cite each factual claim with its context number, for example [Chunk 2]."
)


def generate_answer(question, hits):
    context = "\n\n".join(
        f"[Chunk {number}] (page {chunk['page']}) {chunk['text']}"
        for number, (_, chunk) in enumerate(hits, start=1)
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)["message"]["content"].strip()
    except URLError as error:
        raise SystemExit(f"Local Ollama request failed: {error}. Start Ollama and check the model name.") from error
