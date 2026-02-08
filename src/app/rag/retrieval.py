from typing import List, Tuple, Dict, Any
from app.core.config import settings


def build_context(snippets: List[Tuple[Dict[str, Any], float]]) -> str:
    # Create a compact context block
    parts = []
    for doc, score in snippets:
        parts.append(
            f"[{doc['chunk_id']} | source={doc['source_id']} | score={score:.3f}]\n{doc['text']}\n"
        )
    return "\n".join(parts).strip()


def enough_context(context: str) -> bool:
    return len(context) >= settings.min_context_chars
