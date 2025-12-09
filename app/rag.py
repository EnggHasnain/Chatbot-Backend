from pathlib import Path
import json
from typing import List, Dict, Any

from .ingest import INDEX_FILE # Import INDEX_FILE from .ingest

def retrieve_context(question: str, k: int = 3) -> str:
    """
    Performs a very simple keyword-based retrieval from the indexed documents.
    Returns a single string joining the top-k matching document chunks as context.
    """
    if not INDEX_FILE.exists():
        return ""

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    question_words = set(word.lower() for word in question.split() if word)
    
    scored_documents = []
    for doc in documents:
        text_lower = doc.get("text", "").lower()
        score = sum(1 for word in question_words if word in text_lower)
        if score > 0:
            scored_documents.append({"score": score, **doc})
            
    # Sort by score in descending order
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top k docs with score > 0
    context_parts = []
    for doc in scored_documents:
        if doc["score"] > 0 and len(context_parts) < k:
            # Truncate text to 800 characters
            truncated_text = doc.get("text", "")[:800]
            context_parts.append(truncated_text)
    
    return "\n\n".join(context_parts)