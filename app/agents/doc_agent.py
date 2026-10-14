"""문서 Agent: 감독이 정리한 검색 문장으로 보고서(RAG)를 검색한다."""
from app.agents.state import AgentState
from app.rag.retriever import search_docs


def doc_agent_node(state: AgentState) -> dict:
    query = state["plan"].get("doc_query") or state["question"]
    result = search_docs(query, top_k=4)
    if not result.get("ok"):
        return {"docs": [], "trace": [f"[문서] 검색 실패: {result.get('error')}"]}
    if not result.get("found"):
        return {"docs": [], "trace": [f"[문서] '{query}' 관련 문서 없음"]}
    docs = result["results"]
    pages = ", ".join(f"{d['source']} p.{d['page']}({d['similarity']})" for d in docs)
    return {"docs": docs, "trace": [f"[문서] '{query}' → {pages}"]}
