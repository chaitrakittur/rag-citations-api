from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IngestRequest(BaseModel):
    source_id: str = Field(..., description="Identifier like filename/url/doc_id")
    text: str = Field(..., min_length=1, max_length=1_000_000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source_id: str
    chunks_added: int


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    score: float
    quote: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5_000)
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    used_context: bool
    citations: List[Citation]
    refusal_reason: Optional[str] = None
