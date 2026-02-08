import re
from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        # Try to end at sentence boundary (light heuristic)
        if end < n:
            last_period = chunk.rfind(". ")
            if last_period > 200:
                end = start + last_period + 1
                chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def build_chunk_records(source_id: str, chunks: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for i, c in enumerate(chunks):
        records.append({
            "chunk_id": f"{source_id}::chunk_{i+1}",
            "source_id": source_id,
            "text": c,
            "metadata": metadata or {},
        })
    return records
