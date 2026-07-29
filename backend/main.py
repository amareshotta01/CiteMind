"""
CiteMind backend — FastAPI app.
Both /upload and /query are now real.
"""

import uuid
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from backend.db.mongo import get_db
from backend.providers.router import get_provider
from backend.ingest.loaders import load_document
from backend.ingest.chunker import chunk_pages
from backend.ingest.embed import embed_chunks, build_faiss_index
from backend.ingest.index_store import save_index
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rerank import rerank
from backend.agents.synthesizer import synthesize_answer
from backend.agents.conflict_checker import check_conflicts
from backend.structured.table_qa import store_table, has_table, query_table

app = FastAPI(title="CiteMind API")


class QueryRequest(BaseModel):
    notebook_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]


@app.get("/health")
def health():
    db = get_db()
    return {"status": "ok", "mongo_connected": db is not None}


@app.post("/notebooks")
def create_notebook(name: str):
    db = get_db()
    notebook = {"name": name, "documents": []}
    result = db.notebooks.insert_one(notebook)
    return {"notebook_id": str(result.inserted_id), "name": name}


@app.post("/upload")
async def upload_document(notebook_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    doc_id = str(uuid.uuid4())
    doc = load_document(file.filename, contents)

    if doc["type"] == "pdf":
        chunks = chunk_pages(doc_id=doc_id, pages=doc["content"])
        provider = get_provider()
        embeddings = embed_chunks(chunks, provider)
        index, id_map = build_faiss_index(chunks, embeddings)
        save_index(notebook_id, index, id_map)

        db = get_db()
        db.notebooks.update_one(
            {"_id": __import__("bson").ObjectId(notebook_id)},
            {"$push": {"documents": {"doc_id": doc_id, "filename": file.filename, "chunk_count": len(chunks)}}},
        )

        return {
            "notebook_id": notebook_id,
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": len(doc["content"]),
            "chunks": len(chunks),
            "status": "indexed",
        }

    elif doc["type"] == "csv":
        store_table(notebook_id, file.filename, doc["content"])
        return {
            "notebook_id": notebook_id,
            "filename": file.filename,
            "rows": len(doc["content"]),
            "columns": list(doc["content"].columns),
            "status": "table stored",
        }


@app.post("/query", response_model=QueryResponse)
def query_notebook(req: QueryRequest):
    if has_table(req.notebook_id):
        result = query_table(req.notebook_id, req.question)
        return QueryResponse(**result)

    provider = get_provider()
    candidates = hybrid_search(req.notebook_id, req.question, provider, top_k=20)
    top_chunks = rerank(req.question, candidates, top_k=5)

    conflict_result = check_conflicts(top_chunks)
    result = synthesize_answer(req.question, top_chunks)

    if conflict_result["has_conflict"]:
        result["answer"] = f"⚠️ Note: sources disagree on this — {conflict_result['details']}\n\n{result['answer']}"

    return QueryResponse(**result)