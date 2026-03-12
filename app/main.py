
from app.indexing import (
    load_chunks,
    embed_texts,
    build_faiss_index,
    save_index,
    save_metadata,
    load_index,
    load_metadata,
    search_index,
    answer_with_context,
)
from pathlib import Path
from app.indexing import load_chunks, embed_texts, build_faiss_index, save_index, save_metadata
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import time
import json
import hashlib
from typing import List
from pypdf import PdfReader

from app.chunking import chunk_text
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Booliant RAG Prototype")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

class Health(BaseModel):
    status: str
    env: str

class IngestedFile(BaseModel):
    filename: str
    saved_as: str
    size_bytes: int

class IngestResponse(BaseModel):
    status: str
    files: List[IngestedFile]

class BuildChunksRequest(BaseModel):
    saved_as: str
    chunk_size: int = 1200
    overlap: int = 200

class BuildChunksResponse(BaseModel):
    status: str
    file_id: str
    chunks_path: str
    num_chunks: int

@app.get("/health", response_model=Health)
def health():
    env = os.getenv("APP_ENV", "local")
    return Health(status="ok", env=env)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ingested: List[IngestedFile] = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )

        content = await f.read()
        size = len(content)
        if size == 0:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")

        safe_name = Path(f.filename).name.replace(" ", "_")
        target_name = f"{int(time.time())}_{safe_name}"
        target_path = UPLOADS_DIR / target_name
        target_path.write_bytes(content)

        ingested.append(
            IngestedFile(
                filename=f.filename,
                saved_as=str(target_path),
                size_bytes=size
            )
        )

    return IngestResponse(status="ok", files=ingested)

def _file_id_from_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]

def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(errors="ignore")
    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n\n".join(pages)

    raise HTTPException(status_code=400, detail=f"Unsupported file type for extraction: {ext}")

@app.post("/build_chunks", response_model=BuildChunksResponse)
def build_chunks(req: BuildChunksRequest):
    path = Path(req.saved_as)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.saved_as}")

    text = _extract_text(path)
    chunks = chunk_text(text, chunk_size=req.chunk_size, overlap=req.overlap)

    file_id = _file_id_from_path(path)
    out_path = CHUNKS_DIR / f"{file_id}.json"

    payload = {
        "file_id": file_id,
        "source_file": str(path),
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
        "num_chunks": len(chunks),
        "chunks": [
            {"text": c, "meta": {"source_file": str(path), "chunk_index": idx}}
            for idx, c in enumerate(chunks)
        ],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return BuildChunksResponse(
        status="ok",
        file_id=file_id,
        chunks_path=str(out_path),
        num_chunks=len(chunks),
    )

class BuildIndexRequest(BaseModel):
    chunks_path: str
    embedding_model: str = "text-embedding-3-small"

class BuildIndexResponse(BaseModel):
    status: str
    file_id: str
    index_path: str
    meta_path: str
    num_vectors: int

@app.post("/build_index", response_model=BuildIndexResponse)
def build_index(req: BuildIndexRequest):
    chunks_path = Path(req.chunks_path)
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail=f"Chunks file not found: {req.chunks_path}")

    file_id, chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    meta = [{**c["meta"], "text": c["text"]} for c in chunks]

    vectors = embed_texts(texts, model=req.embedding_model)
    index = build_faiss_index(vectors)

    index_path = INDEX_DIR / f"{file_id}.faiss"
    meta_path = INDEX_DIR / f"{file_id}.meta.json"

    save_index(index, index_path)
    save_metadata(meta, meta_path)

    return BuildIndexResponse(
        status="ok",
        file_id=file_id,
        index_path=str(index_path),
        meta_path=str(meta_path),
        num_vectors=len(texts),
    )

class AskRequest(BaseModel):
    file_id: str
    question: str
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-4.1-mini"

class Citation(BaseModel):
    source_file: str
    chunk_index: int
    score: float
    text: str

class AskResponse(BaseModel):
    status: str
    answer: str
    citations: List[Citation]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    index_path = INDEX_DIR / f"{req.file_id}.faiss"
    meta_path = INDEX_DIR / f"{req.file_id}.meta.json"

    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Index not found: {index_path}")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Metadata not found: {meta_path}")

    index = load_index(index_path)
    metadata = load_metadata(meta_path)

    results = search_index(
        query=req.question,
        index=index,
        metadata=metadata,
        embedding_model=req.embedding_model,
        top_k=req.top_k,
    )

    answer = answer_with_context(
        question=req.question,
        contexts=results,
        model=req.answer_model,
    )

    citations = [
    Citation(
        source_file=r["source_file"],
        chunk_index=r["chunk_index"],
        score=r["score"],
        text=r.get("text", "")
    )
    for r in results
]

    return AskResponse(
        status="ok",
        answer=answer,
        citations=citations,
    )