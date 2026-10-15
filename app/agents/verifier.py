"""검증 단계: 답변에 나온 숫자가 실제 근거(도구 결과·문서)에 있는지 코드로 대조한다.

LLM에게 "검증해 줘"라고 맡기지 않고 코드로 확인한다.
LLM이 LLM을 검증하면 같은 실수를 반복할 수 있기 때문이다 (보내드림의 '결정론적 재검증' 원칙).
"""
import json
import re

from app.agents.state import AgentState
from app.agents.writer import STORE_DATA_SOURCE

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MAX_RETRIES = 1


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in NUMBER.findall(text):
        value = raw.replace(",", "")
        try:
            out.add(str(float(value)))   # "105", "105.0", "1,248" → 같은 값으로 비교
        except ValueError:
            continue
    return out


def _evidence_numbers(state: AgentState) -> set[str]:
    evidence = [state["question"], json.dumps(state.get("plan") or {}, ensure_ascii=False),
                json.dumps(state.get("facts") or [], ensure_ascii=False)]
    if state.get("facts"):
        # 출처 표기 속 날짜(2026.06)를 '근거 없는 숫자'로 오판하지 않게 한다.
        evidence.append(STORE_DATA_SOURCE)
    for d in state.get("docs") or []:
        evidence += [d["source"], str(d["page"]), d["content"]]
    return _numbers(" ".join(evidence))


def verify_node(state: AgentState) -> dict:
    allowed = _evidence_numbers(state)
    # 0~10 같은 작은 정수는 "1년", "3곳 중" 같은 서술에 흔해서 검사에서 제외한다.
    used = {n for n in _numbers(state["draft"]) if float(n) > 10 or "." in n and not n.endswith(".0")}
    unsupported = sorted(used - allowed, key=float)

    if not unsupported:
        return {"answer": state["draft"], "unsupported": [],
                "trace": ["[검증] 답변 속 숫자가 모두 근거와 일치"]}

    retries = state.get("retries", 0)
    if retries < MAX_RETRIES:
        return {"retries": retries + 1, "unsupported": unsupported,
                "feedback": f"다음 숫자는 근거에 없습니다: {unsupported}. 근거에 있는 값으로 고치거나 문장에서 빼세요.",
                "trace": [f"[검증] 근거 없는 숫자 {unsupported} → 재작성 요청"]}

    note = f"\n\n※ 다음 수치는 근거 자료에서 확인되지 않았습니다: {', '.join(unsupported)}"
    return {"answer": state["draft"] + note, "unsupported": unsupported,
            "trace": [f"[검증] 재작성 후에도 근거 없는 숫자 {unsupported} → 경고 표시"]}


def route_after_verify(state: AgentState) -> str:
    return "done" if state.get("answer") else "retry"
