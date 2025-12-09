import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# use absolute imports when running `uvicorn app.main:app`
from app.ingest import ingest_book
from app.rag import retrieve_context
from app.agent import run_agent  # <-- async function

app = FastAPI(title="Physical AI Book Agent")

# CORS configuration for local dev and Vercel production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    print("[HEALTH] ping")
    return {"status": "ok"}


@app.post("/api/ingest")
def api_ingest():
    print("[INGEST] start")
    result = ingest_book()
    print(f"[INGEST] done: {result}")
    return result


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    print(f"[ASK] question={req.question!r}")

    try:
        # sync RAG context retrieval
        context = retrieve_context(req.question)

        if context.strip():
            print("[ASK] context found, calling agent...")
        else:
            print("[ASK] no context found, calling agent anyway for smart response...")

        # Always call agent - it handles greetings, no-context, and context-based answers
        answer = await run_agent(req.question, context)

        print("[ASK] answer produced")
        return {"answer": answer}

    except Exception as e:
        print(f"[ASK][ERROR] {repr(e)}")
        import traceback
        traceback.print_exc()
        return {"answer": "Error: backend failed while calling the model. Check server logs."}



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
