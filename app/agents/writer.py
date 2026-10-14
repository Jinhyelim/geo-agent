"""작성 Agent: 데이터 Agent의 숫자와 문서 Agent의 근거를 합쳐 답변을 쓴다."""
import logging

from app.agents.llm import MODEL, client
from app.agents.state import AgentState

log = logging.getLogger(__name__)

SYSTEM = """너는 서울 종로구 상권 분석 답변 작성 Agent다.
[데이터]와 [문서]에 있는 내용만 근거로 한국어로 답한다.

규칙
- 숫자는 [데이터]나 [문서]에 그대로 있는 값만 쓴다. 새로 계산하거나 추측한 숫자를 만들지 않는다.
- 문서의 숫자는 원문 문장이 말한 의미 그대로만 쓴다. 원문에 없는 비교(예: 서로 다른 업종끼리 비교)를 새로 만들지 않는다.
  특히 '(1.0%p)'처럼 괄호 안 값은 대부분 '서울시와의 차이'이므로 다른 항목과 비교하지 않는다.
- [데이터]에 out_of_service_area가 있으면 그 장소는 종로구 밖이라 분석할 수 없다고 안내한다.
- 문서 내용을 쓰면 문장 끝에 (출처: 파일명 p.쪽) 형식으로 표시한다.
- [데이터]의 숫자를 쓰면 (출처: 소상공인시장진흥공단 상가정보 2026.06 기준)이라고 표시한다.
- 근거가 부족하면 "제공된 데이터로는 확인할 수 없습니다"라고 솔직하게 말한다.
- 4~8문장으로 핵심만 쓰고, 창업 결정을 대신하지 말고 판단에 필요한 정보를 정리한다."""


def _describe_fact(f: dict) -> str | None:
    """도구 결과(JSON)를 의미가 분명한 문장으로 바꾼다.

    처음에는 JSON을 그대로 넘겼는데, LLM이 '반경 안 업종 순위'를 '종로구 전체 순위'로
    잘못 읽었다. 범위(반경 몇 m 안인지)를 문장에 명시해 해석 오류를 줄인다.
    """
    r = f["result"]
    if f["tool"] == "geocode":
        if r.get("ok"):
            return f"- 위치 확인: {r['name']} ({r['address']})"
        if r.get("error") == "out_of_service_area":
            return f"- {r['message']}"
        return f"- '{f['args'].get('query')}' 위치를 찾지 못함"
    if f["tool"] == "count_stores_nearby" and r.get("ok"):
        label = r.get("category") or "전체 업종"
        tops = ", ".join(f"{t['category']} {t['count']}곳" for t in r["top_categories_in_radius"])
        nearest = ", ".join(f"{n['name']}({n['distance_m']}m)" for n in r["nearest"][:3])
        return (f"- 반경 {r['radius_m']}m 안 '{label}' 점포 수: {r['count']}곳\n"
                f"- 같은 반경 {r['radius_m']}m 안 업종 상위(종로구 전체 아님): {tops}\n"
                f"- 가장 가까운 '{label}' 점포: {nearest}")
    return None


def _format_context(state: AgentState) -> str:
    facts = state.get("facts") or []
    docs = state.get("docs") or []
    lines = [d for d in (_describe_fact(f) for f in facts) if d]
    data_part = "\n".join(lines) if lines else "(없음)"
    doc_part = "\n".join(f"- {d['source']} p.{d['page']}: {d['content']}" for d in docs) or "(없음)"
    return f"[데이터]\n{data_part}\n\n[문서]\n{doc_part}"


def _fallback_answer(state: AgentState) -> str:
    # LLM 작성이 실패해도 도구가 계산한 숫자는 그대로 전달한다.
    lines = ["(답변 생성에 실패해 조회 결과만 보여드립니다.)"]
    for f in state.get("facts") or []:
        r = f["result"]
        if f["tool"] == "count_stores_nearby" and r.get("ok"):
            lines.append(f"- 반경 {r['radius_m']}m 안 {r.get('category') or '전체'} 점포: {r['count']}곳")
    for d in state.get("docs") or []:
        lines.append(f"- 참고 문서: {d['source']} p.{d['page']}")
    return "\n".join(lines)


def writer_node(state: AgentState) -> dict:
    user = f"질문: {state['question']}\n\n{_format_context(state)}"
    if state.get("feedback"):
        user += f"\n\n[검증 피드백 - 반드시 반영]\n{state['feedback']}"
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        )
        draft = resp.choices[0].message.content.strip()
        note = "재작성" if state.get("feedback") else "작성"
    except Exception as e:  # noqa: BLE001
        log.warning("[writer] 작성 실패 → 조회 결과만 전달: %s", e)
        draft, note = _fallback_answer(state), "fallback"
    return {"draft": draft, "trace": [f"[작성] 답변 {note} 완료 ({len(draft)}자)"]}
