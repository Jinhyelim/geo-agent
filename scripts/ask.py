"""Multi-Agent에게 질문하고 진행 기록(trace)과 답변을 출력한다.

사용법:
    python -m scripts.ask "광화문 근처에서 카페 창업하면 경쟁이 어때?"
"""
import sys

from app.agents.graph import ask

DEFAULT_QUESTIONS = [
    "광화문 근처에서 카페 창업하면 경쟁이 어때?",
    "종로구 외식업 생존율은 어때?",
    "강남역 맛집 추천해줘",
]

if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    for q in questions:
        print("=" * 70)
        print("질문:", q)
        result = ask(q)
        print("\n[진행 기록]")
        for line in result.get("trace", []):
            print(" ", line)
        print("\n[답변]\n" + result.get("answer", "(없음)"))
        print()
