"""도구 ① 지오코딩: 장소 이름이나 주소를 위도·경도(4326)로 바꾼다.

조회 순서 (질문 형태에 따라 다르게)
  - 행정동 이름("사직동")      → DB의 행정동 중심 좌표를 먼저 쓴다.
                                 카카오 키워드 검색은 "사직동"에 엉뚱한 장소(경희궁)를 돌려준 적이 있다.
  - 주소("세종대로 172")       → 카카오 주소 검색 → 키워드 검색
                                 키워드 검색을 먼저 하면 "세종대로 지하 172"인 역이 잡힌 적이 있다.
  - 그 밖의 장소 이름("광화문") → 카카오 키워드 검색 → 주소 검색
  - 외부 API가 모두 실패하면    → 행정동 중심 좌표로 근사 (fallback)

외부 API 원칙 (보내드림 프로젝트와 동일)
  - timeout 3초
  - 네트워크 오류·5xx는 재시도, 4xx(키 오류·권한 없음)는 재시도해도 같으므로 바로 다음 단계로
  - 예외를 밖으로 던지지 않고 {"ok": False, "error": ...} 로 돌려준다
    → Agent(LLM)가 실패 사실을 보고 다른 방법을 고를 수 있게
"""
import logging
import re

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.db import get_conn

log = logging.getLogger(__name__)

KAKAO_BASE = "https://dapi.kakao.com/v2/local/search"

# 종로구 대략 범위(경도/위도). 키워드 검색 결과를 이 범위 안으로 좁힌다.
# 같은 이름의 장소가 다른 지역에도 있을 수 있기 때문이다 (예: "중앙시장").
JONGNO_RECT = "126.94,37.56,127.02,37.63"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(settings.http_max_attempts),
    wait=wait_exponential(multiplier=0.3, max=2),
    reraise=True,
)
def _kakao_get(path: str, params: dict) -> list[dict]:
    resp = httpx.get(
        f"{KAKAO_BASE}/{path}",
        params=params,
        headers={"Authorization": f"KakaoAK {settings.kakao_rest_api_key}"},
        timeout=settings.http_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json().get("documents", [])


# "세종대로 172", "종로1길 50", "율곡로10길 3"처럼 도로명 뒤에 '띄어쓰기 + 건물번호'가 오면 주소로 본다.
# 띄어쓰기를 필수로 한 이유: "종로3가역"의 '종로3'이 주소로 잘못 판정된 적이 있다.
ADDRESS_PATTERN = re.compile(r"(로|길)\s+\d")


def looks_like_address(query: str) -> bool:
    return bool(ADDRESS_PATTERN.search(query))


def _from_kakao(query: str) -> dict | None:
    if not settings.kakao_rest_api_key:
        return None
    keyword = ("keyword.json", {"query": query, "rect": JONGNO_RECT, "size": 1})
    address = ("address.json", {"query": query, "size": 1})
    attempts = [address, keyword] if looks_like_address(query) else [keyword, address]
    for path, params in attempts:
        try:
            docs = _kakao_get(path, params)
        except Exception as e:  # noqa: BLE001 - 어떤 실패든 다음 단계로 넘어간다
            log.warning("[geocode] kakao %s 실패: %s", path, e)
            continue
        if docs:
            d = docs[0]
            return {
                "ok": True,
                "lat": float(d["y"]),   # 카카오는 x=경도, y=위도 (4326)
                "lon": float(d["x"]),
                "name": d.get("place_name") or d.get("address_name"),
                "address": d.get("road_address_name") or d.get("address_name"),
                "source": f"kakao:{path.split('.')[0]}",
            }
    return None


def _from_dong_centroid(query: str, exact: bool = False) -> dict | None:
    """행정동 이름이면 그 동 점포들의 평균 좌표를 중심점으로 쓴다.

    exact=True  : 행정동 이름과 정확히 같을 때만 (질문이 행정동 이름인지 판정할 때)
    exact=False : 앞부분만 같아도 허용 (외부 API가 모두 실패했을 때의 fallback)
    """
    condition = "dong = %(q)s" if exact else "(dong = %(q)s OR dong LIKE %(q)s || '%%')"
    sql = f"""
        SELECT dong, AVG(ST_Y(geom::geometry)) AS lat, AVG(ST_X(geom::geometry)) AS lon
        FROM stores
        WHERE {condition}
        GROUP BY dong
        ORDER BY count(*) DESC
        LIMIT 1
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, {"q": query.strip()})
        row = cur.fetchone()
    if not row:
        return None
    return {
        "ok": True,
        "lat": float(row[1]),
        "lon": float(row[2]),
        "name": row[0],
        "address": f"서울 종로구 {row[0]} (행정동 중심, 근사값)",
        "source": "db:dong_centroid" if exact else "fallback:dong_centroid",
    }


def geocode(query: str) -> dict:
    if not query or not query.strip():
        return {"ok": False, "error": "empty_query"}

    result = (
        _from_dong_centroid(query, exact=True)   # 1) 행정동 이름이면 DB 기준 좌표
        or _from_kakao(query)                    # 2) 카카오 (주소/키워드 순서는 질문 형태로 결정)
        or _from_dong_centroid(query)            # 3) 외부 API 실패 시 근사 fallback
    )
    if result:
        return result
    return {"ok": False, "error": "not_found", "query": query}
