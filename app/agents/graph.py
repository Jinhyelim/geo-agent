"""Multi-Agent Workflow 정의 (LangGraph).

    질문 → [감독] ─┬─▶ [데이터] ─┬─▶ [문서] ─▶ [작성] ─▶ [검증] ─▶ 답변
                    │             └──────────────▲          │
                    └─▶ [문서] ──────────────────┘          └─(근거 없는 숫자)─▶ [작성] 1회 재작성
"""
from langgraph.graph import END, StateGraph

from app.agents.data_agent import data_agent_node
from app.agents.doc_agent import doc_agent_node
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.verifier import route_after_verify, verify_node
from app.agents.writer import writer_node


def _after_supervisor(state: AgentState) -> str:
    plan = state["plan"]
    if plan.get("needs_data"):
        return "data"
    return "docs" if plan.get("needs_docs") else "write"


def _after_data(state: AgentState) -> str:
    return "docs" if state["plan"].get("needs_docs") else "write"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("data_agent", data_agent_node)
    g.add_node("doc_agent", doc_agent_node)
    g.add_node("writer", writer_node)
    g.add_node("verifier", verify_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", _after_supervisor,
                            {"data": "data_agent", "docs": "doc_agent", "write": "writer"})
    g.add_conditional_edges("data_agent", _after_data, {"docs": "doc_agent", "write": "writer"})
    g.add_edge("doc_agent", "writer")
    g.add_edge("writer", "verifier")
    g.add_conditional_edges("verifier", route_after_verify, {"retry": "writer", "done": END})
    return g.compile()


graph = build_graph()


def ask(question: str) -> dict:
    return graph.invoke({"question": question, "trace": []})
