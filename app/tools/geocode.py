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

SERVICE_AREA = "종로구"  # 이 프로젝트의 분석 범위. 상가 데이터도 종로구만 적재되어 있다.

# 처음에는 키워드 검색을 종로구 사각형 범위(rect)로 제한했는데,
# "강남역"처럼 범위 밖 장소를 물으면 범위 안의 엉뚱한 장소가 대신 잡혔다.
# 그래서 범위 제한 없이 검색한 뒤, 결과 주소가 종로구인지 코드로 확인한다.


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
    keyword = ("keyword.json", {"query": query, "size": 5})
    address = ("address.json", {"query": query, "size": 5})
    attempts = [address, keyword] if looks_like_address(query) else [keyword, address]
    outside = None  # 결과는 있었지만 모두 종로구 밖이었던 경우의 주소
    for path, params in attempts:
        try:
            docs = _kakao_get(path, params)
        except Exception as e:  # noqa: BLE001 - 어떤 실패든 다음 단계로 넘어간다
            log.warning("[geocode] kakao %s 실패: %s", path, e)
            continue
        in_area = [d for d in docs if SERVICE_AREA in (d.get("address_name") or "")]
        if docs and not in_area:
            outside = outside or docs[0].get("address_name")
        if in_area:
            d = in_area[0]  # 같은 이름이 여러 지역에 있으면 종로구 것을 고른다 (예: "중앙시장")
            return {
                "ok": True,
                "lat": float(d["y"]),   # 카카오는 x=경도, y=위도 (4326)
                "lon": float(d["x"]),
                "name": d.get("place_name") or d.get("address_name"),
                "address": d.get("road_address_name") or d.get("address_name"),
                "source": f"kakao:{path.split('.')[0]}",
            }
    if outside:
        return {"ok": False, "error": "out_of_service_area", "query": query,
                "found_address": outside,
                "message": f"'{query}'은(는) {SERVICE_AREA} 밖({outside})이라 분석 범위가 아닙니다."}
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
    # 서비스 지역 밖으로 확인된 경우는 fallback으로 덮지 않고 그대로 알린다.
    if result:
        return result
    return {"ok": False, "error": "not_found", "query": query}
