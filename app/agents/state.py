"""Agent들이 함께 읽고 쓰는 공유 상태(State).

LangGraph는 각 Agent(노드)가 이 상태의 일부를 돌려주면 합쳐서 다음 Agent에게 넘긴다.
"""
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    question: str                 # 사용자 질문
    plan: dict                    # 감독 Agent가 세운 계획 (어떤 Agent가 필요한지)
    facts: list[dict]             # 데이터 Agent가 도구로 얻은 결과 (숫자의 근거)
    docs: list[dict]              # 문서 Agent가 찾은 문서 조각 (해석의 근거)
    draft: str                    # 작성 Agent가 쓴 답변
    feedback: str                 # 검증 결과 고쳐야 할 점 (재작성 시 전달)
    retries: int                  # 재작성 횟수
    unsupported: list[str]        # 근거를 찾지 못한 숫자
    answer: str                   # 최종 답변
    trace: Annotated[list[str], operator.add]  # 진행 기록 (각 Agent가 한 줄씩 덧붙임)
