from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_book
from app.rag import retrieve_context
from app.agent import run_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/ingest")
def api_ingest():
    """
    Run ingest_book() and return its dict result directly.
    """
    return ingest_book()

@app.post("/api/ask")
def api_ask(req: AskRequest):
    print(f"[ASK] question={req.question!r}") # Added print logging
    context = retrieve_context(req.question)

    if not context.strip():
        return {"answer": "This information is not found in the textbook."}

    answer = run_agent(req.question, context)
    return {"answer": answer}

    