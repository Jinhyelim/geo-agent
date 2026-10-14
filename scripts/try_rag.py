"""문서 검색(RAG) 도구를 단독으로 실행해 보는 스크립트.

사용법:
    python -m scripts.try_rag
"""
from app.rag.retriever import search_docs

QUESTIONS = [
    "종로구에서 발달상권은 어디야?",
    "종로구 소상공인은 SNS 마케팅을 많이 해?",
    "종로구 외식업 생존율은 어때?",
    "강남역 맛집 추천해줘",   # 문서에 없는 질문 → 근거 없음이 나와야 정상
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"\n=== {q}")
        r = search_docs(q, top_k=3)
        if not r.get("found"):
            print("  →", r)
            continue
        for h in r["results"]:
            print(f"  [{h['similarity']}] {h['source']} p.{h['page']}: {h['content'][:110]}")
