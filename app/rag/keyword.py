"""키워드 검색 (BM25) — 하이브리드 검색의 한쪽 축.

벡터 검색은 '뜻이 비슷한' 조각을 잘 찾지만, '광화문'처럼 정확한 고유명사가
문서에 그대로 적힌 조각은 놓칠 때가 있다. BM25는 질문의 단어가 조각에
'실제로 몇 번, 얼마나 드물게' 나오는지로 점수를 매겨 이 빈틈을 메운다.

한국어는 띄어쓰기 단위로 자르면 '광화문역은', '광화문역이'가 서로 다른 단어가 되므로
형태소 분석기(kiwipiepy)로 명사·외국어·숫자 등 뜻을 가진 부분만 뽑아 쓴다.
"""
import re
from functools import lru_cache

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from app.core.db import get_conn

# 뜻을 가진 형태소만 사용: 일반/고유명사(NNG, NNP), 외국어(SL), 숫자(SN), 어근(XR)
KEEP_TAGS = {"NNG", "NNP", "SL", "SN", "XR"}

def _load_dong_names() -> list[str]:
    """DB에 적재된 종로구 행정동 이름 (예: 창신2동, 종로1·2·3·4가동)."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT dong FROM stores WHERE dong IS NOT NULL")
            return [r[0] for r in cur.fetchall()]
    except Exception:  # noqa: BLE001 - DB가 없어도 검색은 동작하게 (사전 없이 진행)
        return []


def _build_kiwi() -> Kiwi:
    """형태소 분석기 + 사용자 사전.

    사전 없이 나누면 "창신2동"이 "창신 / 2 / 동"으로 쪼개진다. "2"와 "동"은 거의 모든 청크에 있어
    변별력이 없고, "창신"은 창신1·2·3동을 구분하지 못해 정답 청크가 키워드 검색 54위로 밀렸다.
    그래서 행정동 이름을 '쪼개지 않는 고유명사'로 등록한다. 질문과 문서를 같은 사전으로 나눈다.
    """
    kiwi = Kiwi()
    for name in _load_dong_names():
        # DB는 "종로5.6가동"(마침표), 보고서는 "종로5·6가동"(가운뎃점)이라 두 표기를 모두 등록
        for variant in {name, name.replace(".", "·")}:
            kiwi.add_user_word(variant, "NNP", 10.0)
    return kiwi


_kiwi = _build_kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form for t in _kiwi.tokenize(text) if t.tag in KEEP_TAGS]


def proper_nouns(text: str) -> list[str]:
    """고유명사(NNP)만. 예: '광화문 상권 특징은?' → ['광화문']"""
    return [t.form for t in _kiwi.tokenize(text) if t.tag == "NNP"]


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[tuple]]:
    """doc_chunks 전체로 BM25 색인을 메모리에 만든다 (첫 호출 때 한 번만).

    조각이 186개뿐이라 메모리 색인으로 충분하다. 문서가 수만 개로 늘면
    PostgreSQL 전문 검색이나 Elasticsearch 같은 별도 색인이 필요하다.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, source, page, content FROM doc_chunks ORDER BY id")
        rows = cur.fetchall()
    corpus = [tokenize(r[3]) for r in rows]
    return BM25Okapi(corpus), rows


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset[str]:
    bm25, _ = _index()
    return frozenset(t for doc in bm25.doc_freqs for t in doc)


def in_corpus(term: str) -> bool:
    """이 단어가 문서 어딘가에 '단어로' 나오는지.

    - 글자 포함(term in 원문)만 보면 "부산"이 "세부산출내역" 안에 있어 오판했다.
    - 형태소 단어 목록만 보면 "세종마을음식문화거리"를 다르게 잘라 "세종"을 놓쳤다.
    → 단어 목록에 있거나, 원문에서 '단어의 시작'(앞에 한글이 없는 위치)으로 나오면 있다고 본다.
    """
    if term in _vocabulary():
        return True
    pattern = re.compile(rf"(?<![가-힣]){re.escape(term)}")
    _, rows = _index()
    return any(pattern.search(r[3]) for r in rows)


def keyword_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 점수 상위 조각. 질문 단어가 하나도 안 겹치는 조각(점수 0)은 뺀다."""
    bm25, rows = _index()
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"id": rows[i][0], "source": rows[i][1], "page": rows[i][2],
         "content": rows[i][3], "bm25": round(float(scores[i]), 2)}
        for i in order if scores[i] > 0
    ]
