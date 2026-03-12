from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]

def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for a prototype; we can upgrade to token-based later.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks
