from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import os

import faiss
import numpy as np
from openai import OpenAI

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

def load_chunks(chunks_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    payload = json.loads(chunks_path.read_text())
    file_id = payload["file_id"]
    chunks = payload["chunks"]
    return file_id, chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    client = _get_client()
    res = client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in res.data], dtype="float32")
    return vectors

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

def save_metadata(meta: List[Dict[str, Any]], meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))

def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    return json.loads(meta_path.read_text())

def search_index(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    client = _get_client()
    res = client.embeddings.create(model=embedding_model, input=[query])
    qvec = np.array([res.data[0].embedding], dtype="float32")
    faiss.normalize_L2(qvec)

    scores, indices = index.search(qvec, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)

    return results

def answer_with_context(
    question: str,
    contexts: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini"
) -> str:
    client = _get_client()

    context_text = "\n\n".join(
        [
            f"[Source {i+1}] chunk_index={c.get('chunk_index')} source_file={c.get('source_file')}\n{c.get('text', '')}"
            for i, c in enumerate(contexts)
        ]
    )

    prompt = f"""
You are answering questions using only the provided context from enterprise documents.

Rules:
- Use only the provided context.
- Do not guess or add outside knowledge.
- If the answer is not clearly supported by the context, say: "The information is not available in the provided documents."
- Be specific.
- Prefer concrete details over general summaries.
- At the end, include a short "Sources" section listing the relevant source numbers used.

Question:
{question}

Context:
{context_text}
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text
