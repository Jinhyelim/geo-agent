"""벡터 검색 vs 하이브리드 검색 비교.

사용법:
    python -m scripts.compare_search
"""
from app.rag.retriever import search_docs

QUESTIONS = [
    "광화문 상권 특징은?",
    "광화문 근처 카페 창업 경쟁은 어때?",
    "종로구 외식업 생존율은 어때?",
    "강남역 맛집 추천해줘",   # 문서에 없는 질문 → 두 방식 모두 근거 없음이어야 정상
]


def show(label: str, r: dict) -> None:
    print(f"  [{label}]")
    if not r.get("found"):
        print("    →", r.get("message", r))
        return
    for h in r["results"]:
        extra = f" bm25={h['bm25']}" if "bm25" in h else ""
        print(f"    sim={h['similarity']}{extra} {h['source']} p.{h['page']}: {h['content'][:70]}")


if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"\n=== {q}")
        show("vector", search_docs(q, top_k=4, mode="vector"))
        show("hybrid", search_docs(q, top_k=4, mode="hybrid"))
