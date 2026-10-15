"""FastAPI 서비스 — Agent를 HTTP API로 제공한다.

실행:
    uvicorn app.main:app --reload
문서(Swagger):
    http://localhost:8000/docs
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.graph import ask
from app.core.db import get_conn
from app.rag.retriever import search_docs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("geo-agent")

app = FastAPI(
    title="geo-agent",
    description="종로구 상권·입지 분석 AI Agent (PostGIS 공간 분석 + 하이브리드 RAG + Multi-Agent)",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=200,
                          examples=["광화문 근처에서 카페 창업하면 경쟁이 어때?"])


class AskResponse(BaseModel):
    answer: str
    plan: dict = {}
    unsupported_numbers: list[str] = []   # 검증에서 근거를 못 찾은 숫자 (비어 있어야 정상)
    trace: list[str] = []                 # Agent 진행 기록 (어떤 Agent가 무엇을 했는지)
    elapsed_ms: int


class DocSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    top_k: int = Field(4, ge=1, le=10)
    mode: str = Field("hybrid", pattern="^(hybrid|vector)$")


@app.get("/health")
def health() -> dict:
    """서버와 DB가 살아 있는지 확인 (운영 시 모니터링·장애 감지용)."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM stores")
            stores = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM doc_chunks")
            chunks = cur.fetchone()[0]
    except Exception as e:  # noqa: BLE001
        log.error("health check failed: %s", e.__class__.__name__)
        raise HTTPException(status_code=503, detail="database_unavailable") from e
    return {"status": "ok", "stores": stores, "doc_chunks": chunks}


# 동기(def) 함수로 두면 FastAPI가 별도 스레드에서 실행해, LLM 응답을 기다리는 동안 다른 요청을 막지 않는다.
@app.post("/ask", response_model=AskResponse)
def ask_agent(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    try:
        state = ask(req.question)
    except Exception as e:  # noqa: BLE001
        log.exception("agent failed: question=%r", req.question)
        raise HTTPException(status_code=503, detail="agent_unavailable") from e
    elapsed = int((time.perf_counter() - started) * 1000)
    log.info("ask done in %dms, unsupported=%s", elapsed, state.get("unsupported", []))
    return AskResponse(
        answer=state.get("answer", ""),
        plan=state.get("plan", {}),
        unsupported_numbers=state.get("unsupported", []),
        trace=state.get("trace", []),
        elapsed_ms=elapsed,
    )


@app.post("/search/docs")
def search_documents(req: DocSearchRequest) -> dict:
    """문서 검색만 따로 호출 (hybrid / vector 비교·디버깅용)."""
    return search_docs(req.query, top_k=req.top_k, mode=req.mode)
