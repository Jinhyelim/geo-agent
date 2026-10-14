"""감독 Agent: 질문을 읽고 어떤 Agent에게 무엇을 맡길지 계획을 세운다."""
import json
import logging

from app.agents.llm import MODEL, client
from app.agents.state import AgentState

log = logging.getLogger(__name__)

SYSTEM = """너는 서울 종로구 상권 분석 서비스의 감독 Agent다.
사용자 질문을 보고 아래 두 담당자 중 누가 필요한지 정하고, 각자에게 넘길 정보를 정리한다.

- 데이터 담당: 특정 장소 주변의 점포 수·업종 분포처럼 '지금 몇 개 있는지' 숫자가 필요할 때
- 문서 담당: 상권의 흐름, 생존율, 개업·폐업, 소상공인 실태처럼 보고서 내용이 필요할 때
- 창업 판단, 경쟁, 전망을 묻는 질문은 '지금의 숫자'와 '상권의 흐름'이 모두 필요하므로 두 담당자를 모두 배정한다.
- 분석 범위는 서울 종로구다. 다른 지역 장소라도 place에 그대로 적는다 (범위 판정은 데이터 담당의 도구가 한다).

반드시 아래 JSON만 출력한다.
{
  "needs_data": true/false,
  "place": "장소 이름 또는 주소 (없으면 null)",
  "category": "업종 이름, 예: 카페, 편의점 (없으면 null)",
  "radius_m": 반경 미터 (기본 500),
  "needs_docs": true/false,
  "doc_query": "보고서에서 찾을 검색 문장 (없으면 null)"
}"""


def _fallback_plan(question: str) -> dict:
    # LLM 호출이 실패해도 서비스가 멈추지 않도록 기본 계획을 쓴다.
    # 장소를 알 수 없으니 데이터 담당은 부르지 않고, 질문 그대로 문서만 검색한다.
    return {"needs_data": False, "place": None, "category": None, "radius_m": 500,
            "needs_docs": True, "doc_query": question, "fallback": True}


def supervisor_node(state: AgentState) -> dict:
    question = state["question"]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},  # JSON 형식 강제
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": question}],
        )
        plan = json.loads(resp.choices[0].message.content)
        # LLM 출력은 신뢰할 수 없는 입력이므로 필요한 값만 꺼내고 형식을 맞춘다.
        plan = {
            "needs_data": bool(plan.get("needs_data")) and bool(plan.get("place")),
            "place": plan.get("place"),
            "category": plan.get("category"),
            "radius_m": int(plan.get("radius_m") or 500),
            "needs_docs": bool(plan.get("needs_docs")),
            "doc_query": plan.get("doc_query") or question,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("[supervisor] 계획 실패 → 기본 계획 사용: %s", e)
        plan = _fallback_plan(question)

    who = [n for n, need in (("데이터", plan["needs_data"]), ("문서", plan["needs_docs"])) if need]
    return {"plan": plan, "retries": 0,
            "trace": [f"[감독] 담당 배정: {', '.join(who) or '없음'} / 계획: {plan}"]}
