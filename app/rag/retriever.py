"""도구 ③ 문서 검색 (RAG의 R = Retrieval) — 하이브리드 검색.

1) 벡터 검색: 질문과 '뜻'이 가까운 조각 (pgvector, 코사인 유사도)
2) 키워드 검색: 질문의 '단어'가 실제로 들어 있는 조각 (BM25)
3) RRF로 두 순위를 합친다.
유사도가 기준보다 낮으면 '근거 없음'으로 처리해, LLM이 문서에 없는 내용을 지어내지 않게 한다.
"""
import numpy as np

from app.core.db import get_conn
from app.rag.embedding import embed_texts
from app.rag.keyword import keyword_search, proper_nouns

# 코사인 유사도 기준. 실제 문서 3개(186조각)로 측정한 값을 보고 정했다.
#   관련 있는 질문: 0.43 ~ 0.59 / 문서에 없는 질문("강남역 맛집"): 0.32 ~ 0.34
# 처음 0.3으로 두었을 때 관련 없는 표 조각까지 '근거'로 통과해서 0.4로 올렸다.
MIN_SIMILARITY = 0.4

# 키워드 검색이 찾은 조각의 완화 기준.
# '광화문'은 문서에서 행정동(사직동)·역 이름의 일부로만 나와서, 표 조각의 유사도가 0.3대에 머문다.
#   예) p.53 "사직동 광화문역 발달상권 2,017 ..." → 유사도 0.315 (벡터 검색은 놓침, BM25는 찾음)
# 그래서 '질문의 고유명사가 조각에 그대로 들어 있으면' 0.3까지 근거로 인정한다.
# 문서에 없는 질문("강남역 맛집")은 고유명사가 어느 조각에도 없어 여전히 걸러진다.
KEYWORD_MIN_SIMILARITY = 0.3

CANDIDATES = 20  # 두 검색에서 각각 가져올 후보 수
RRF_K = 60       # RRF 상수. 순위 1위와 2위의 점수 차이를 완만하게 만든다 (논문 기본값).


def _vector_search(qvec: np.ndarray, limit: int) -> list[dict]:
    sql = """
        SELECT id, source, page, content, 1 - (embedding <=> %s) AS similarity  -- <=> : 코사인 거리
        FROM doc_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (qvec, qvec, limit))
        rows = cur.fetchall()
    return [
        {"id": r[0], "source": r[1], "page": r[2], "content": r[3], "similarity": round(float(r[4]), 3)}
        for r in rows
    ]


def _similarities(qvec: np.ndarray, ids: list[int]) -> dict[int, float]:
    """키워드 검색으로만 찾은 조각도 '뜻이 얼마나 가까운지' 확인하려고 유사도를 계산한다."""
    if not ids:
        return {}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, 1 - (embedding <=> %s) FROM doc_chunks WHERE id = ANY(%s)", (qvec, ids)
        )
        return {r[0]: round(float(r[1]), 3) for r in cur.fetchall()}


def rrf_merge(*rankings: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: 각 검색의 '순위'만 보고 점수 = Σ 1/(K + 순위).

    벡터 유사도(0~1)와 BM25 점수(0~수십)는 단위가 달라 그대로 더할 수 없다.
    순위는 단위가 없으니, 두 검색 모두에서 위에 있는 조각이 자연스럽게 1등이 된다.
    """
    merged: dict[int, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            item = merged.setdefault(hit["id"], {**hit, "rrf": 0.0})
            item.update({k: v for k, v in hit.items() if k not in item})
            item["rrf"] += 1 / (RRF_K + rank)
    return sorted(merged.values(), key=lambda h: h["rrf"], reverse=True)


def search_docs(query: str, top_k: int = 4, mode: str = "hybrid") -> dict:
    """mode: "hybrid"(기본) 또는 "vector"(비교용, 이전 방식)."""
    try:
        qvec = np.array(embed_texts([query])[0])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"embedding_failed: {e.__class__.__name__}"}

    vector_hits = _vector_search(qvec, CANDIDATES)
    if mode == "vector":
        ranked = vector_hits
    else:
        keyword_hits = keyword_search(query, CANDIDATES)
        sims = _similarities(qvec, [h["id"] for h in keyword_hits])
        for h in keyword_hits:
            h["similarity"] = sims.get(h["id"], 0.0)
        ranked = rrf_merge(vector_hits, keyword_hits)

    nouns = proper_nouns(query) if mode != "vector" else []

    def is_evidence(h: dict) -> bool:
        if h["similarity"] >= MIN_SIMILARITY:
            return True
        return h["similarity"] >= KEYWORD_MIN_SIMILARITY and any(n in h["content"] for n in nouns)

    hits = [
        {k: h[k] for k in ("source", "page", "content", "similarity", "bm25", "rrf") if k in h}
        for h in ranked if is_evidence(h)
    ][:top_k]
    if not hits:
        return {"ok": True, "found": False, "message": "관련 문서를 찾지 못했습니다."}
    return {"ok": True, "found": True, "results": hits}
