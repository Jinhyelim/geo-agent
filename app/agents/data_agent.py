"""데이터 Agent: 지오코딩·반경 검색 도구를 스스로 골라 호출해 숫자 근거를 모은다.

OpenAI의 tool calling을 쓴다. LLM은 "어떤 도구를 어떤 값으로 부를지"만 정하고,
실제 계산은 우리 코드(PostGIS)가 한다. 그래서 숫자는 LLM이 지어낸 것이 아니다.
"""
import json
import logging

from app.agents.llm import MODEL, client
from app.agents.state import AgentState
from app.tools.geocode import geocode
from app.tools.spatial import count_stores_nearby

log = logging.getLogger(__name__)
MAX_STEPS = 5  # 도구 호출 반복 상한. LLM이 같은 도구를 끝없이 부르는 것을 막는다.

TOOLS = [
    {"type": "function", "function": {
        "name": "geocode",
        "description": "장소 이름이나 주소를 위도·경도로 바꾼다. 반경 검색 전에 먼저 호출한다.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "예: 광화문, 종로3가역, 세종대로 172, 사직동"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "count_stores_nearby",
        "description": "좌표 반경 안의 점포 수를 업종별로 센다.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
            "radius_m": {"type": "integer", "description": "50~2000"},
            "category": {"type": "string", "description": "업종 이름 (예: 카페). 전체면 생략"}},
            "required": ["lat", "lon"]}}},
]
FUNCTIONS = {"geocode": geocode, "count_stores_nearby": count_stores_nearby}

SYSTEM = """너는 상권 데이터 담당 Agent다. 도구를 사용해 질문에 필요한 숫자를 모은다.
1) geocode로 장소 좌표를 얻고, 2) count_stores_nearby로 반경 안 점포를 센다.
도구가 실패(ok=false)하면 다른 표현으로 한 번 더 시도하고, 그래도 안 되면 실패했다고 보고한다.
단, error가 out_of_service_area면 종로구 밖이라는 뜻이니 재시도하지 말고 그대로 보고한다.
숫자는 절대 추측하지 말고 도구 결과만 사용한다. 마지막에 모은 결과를 한두 문장으로 요약한다."""


def _call_tool(name: str, args: dict) -> dict:
    func = FUNCTIONS.get(name)
    if not func:
        return {"ok": False, "error": f"unknown_tool:{name}"}
    try:
        return func(**args)
    except Exception as e:  # noqa: BLE001 - 도구 오류가 Agent 전체를 멈추지 않게
        log.warning("[data_agent] %s 실패: %s", name, e)
        return {"ok": False, "error": f"{name}_failed:{e.__class__.__name__}"}


def data_agent_node(state: AgentState) -> dict:
    plan = state["plan"]
    hint = (f"장소: {plan.get('place')}, 업종: {plan.get('category') or '전체'}, "
            f"반경: {plan.get('radius_m', 500)}m")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"질문: {state['question']}\n힌트: {hint}"}]
    facts, trace = [], []

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            trace.append(f"[데이터] 요약: {msg.content}")
            break
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = _call_tool(call.function.name, args)
            facts.append({"tool": call.function.name, "args": args, "result": result})
            trace.append(f"[데이터] 도구 호출 {call.function.name}({args}) → ok={result.get('ok')}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    else:
        trace.append(f"[데이터] 도구 호출이 {MAX_STEPS}회를 넘어 중단")

    return {"facts": facts, "trace": trace}
