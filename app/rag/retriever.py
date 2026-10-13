"""도구 ③ 문서 검색 (RAG의 R = Retrieval).

질문을 임베딩해 doc_chunks에서 의미가 가장 가까운 조각을 찾는다.
유사도가 기준보다 낮으면 '근거 없음'으로 처리해, LLM이 문서에 없는 내용을 지어내지 않게 한다.
"""
import numpy as np

from app.core.db import get_conn
from app.rag.embedding import embed_texts

MIN_SIMILARITY = 0.3  # 코사인 유사도 기준. 실제 문서를 넣은 뒤 결과를 보고 조정한다.


def search_docs(query: str, top_k: int = 4) -> dict:
    try:
        qvec = np.array(embed_texts([query])[0])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"embedding_failed: {e.__class__.__name__}"}

    sql = """
        SELECT source, page, content, 1 - (embedding <=> %s) AS similarity  -- <=> : 코사인 거리
        FROM doc_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (qvec, qvec, top_k))
        rows = cur.fetchall()

    hits = [
        {"source": r[0], "page": r[1], "content": r[2], "similarity": round(float(r[3]), 3)}
        for r in rows if r[3] >= MIN_SIMILARITY
    ]
    if not hits:
        return {"ok": True, "found": False, "message": "관련 문서를 찾지 못했습니다."}
    return {"ok": True, "found": True, "results": hits}
