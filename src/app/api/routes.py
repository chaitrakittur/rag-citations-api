from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.services.openai_client import get_openai_client
from app.rag.schemas import IngestRequest, IngestResponse, AskRequest, AskResponse, Citation
from app.rag.chunking import chunk_text, build_chunk_records
from app.rag.store import VectorStore
from app.rag.retrieval import build_context, enough_context
from app.rag.prompts import SYSTEM_PROMPT

router = APIRouter()
store = VectorStore()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    records = build_chunk_records(req.source_id, chunks, req.metadata)

    if not records:
        return IngestResponse(source_id=req.source_id, chunks_added=0)

    client = get_openai_client()
    try:
        emb_resp = client.embeddings.create(
            model=settings.openai_embed_model,
            input=[r["text"] for r in records],
        )
        embeddings = [e.embedding for e in emb_resp.data]
        store.add(records, embeddings)
        return IngestResponse(source_id=req.source_id, chunks_added=len(records))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    top_k = req.top_k or settings.top_k
    client = get_openai_client()

    try:
        q_emb = client.embeddings.create(
            model=settings.openai_embed_model,
            input=req.question,
        ).data[0].embedding

        hits = store.search(q_emb, top_k=top_k)
        context = build_context(hits)

        citations = [
            Citation(
                chunk_id=doc["chunk_id"],
                source_id=doc["source_id"],
                score=score,
                quote=(doc["text"][:260] + "…") if len(doc["text"]) > 260 else doc["text"],
            )
            for doc, score in hits
        ]

        # Guardrail: if context is too small/empty, refuse.
        if not enough_context(context):
            return AskResponse(
                answer="I don’t know based on the provided documents.",
                used_context=False,
                citations=citations,
                refusal_reason="insufficient_context",
            )

        # Use Responses API
        resp = client.responses.create(
            model=settings.openai_chat_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{req.question}\n\nCONTEXT:\n{context}"},
            ],
        )

        answer = resp.output_text.strip() if hasattr(resp, "output_text") else str(resp)
        # Extra safety: if model tries to answer without context
        if answer.lower().startswith(("i don't know", "i do not know")):
            return AskResponse(answer=answer, used_context=True, citations=citations, refusal_reason="model_refused")

        return AskResponse(answer=answer, used_context=True, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}")
