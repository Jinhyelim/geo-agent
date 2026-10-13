"""도구 ② 반경 내 점포 검색 (PostGIS).

특정 좌표 반경 N미터 안에 있는 점포 수를 업종별로 센다.
Agent가 "광화문 근처 카페 경쟁이 어때?"에 답할 때 쓰는 핵심 숫자를 만든다.

이 함수가 돌려주는 숫자는 LLM이 지어낸 것이 아니라 DB에서 계산한 '정답'이다.
나중에 검증 단계에서 LLM 답변 속 숫자를 이 값과 대조한다.
"""
from app.core.db import get_conn

MIN_RADIUS_M = 50
MAX_RADIUS_M = 2000


def count_stores_nearby(lat: float, lon: float, radius_m: int = 500,
                        category: str | None = None, sample_size: int = 5) -> dict:
    # LLM이 이상한 반경(예: 0, 100km)을 넘겨도 DB가 과부하되지 않게 범위를 제한한다.
    radius_m = max(MIN_RADIUS_M, min(int(radius_m), MAX_RADIUS_M))

    # 기준점: 경도가 먼저(ST_MakePoint(경도, 위도)), 4326 명시
    point = "ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography"

    # 업종 필터: 대·중·소분류 중 어디든 일치하면 포함 (예: "카페"=소분류, "음식"=대분류)
    category_filter = ""
    params = {"lat": lat, "lon": lon, "r": radius_m}
    if category:
        category_filter = """
            AND (category_small = %(cat)s OR category_mid = %(cat)s OR category_large = %(cat)s)
        """
        params["cat"] = category.strip()

    count_sql = f"""
        SELECT count(*)
        FROM stores
        WHERE ST_DWithin(geom, {point}, %(r)s)   -- 반경 안인지 판정 (GIST 인덱스 사용)
        {category_filter}
    """
    sample_sql = f"""
        SELECT name, category_small, road_address,
               round(ST_Distance(geom, {point})::numeric) AS distance_m
        FROM stores
        WHERE ST_DWithin(geom, {point}, %(r)s)
        {category_filter}
        ORDER BY distance_m
        LIMIT %(n)s
    """
    breakdown_sql = f"""
        SELECT category_small, count(*) AS cnt
        FROM stores
        WHERE ST_DWithin(geom, {point}, %(r)s)
        GROUP BY category_small
        ORDER BY cnt DESC
        LIMIT 5
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = cur.fetchone()[0]

        cur.execute(sample_sql, {**params, "n": sample_size})
        nearest = [
            {"name": r[0], "category": r[1], "address": r[2], "distance_m": int(r[3])}
            for r in cur.fetchall()
        ]

        cur.execute(breakdown_sql, params)
        top_categories = [{"category": r[0], "count": r[1]} for r in cur.fetchall()]

    return {
        "ok": True,
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "category": category,
        "count": total,
        "nearest": nearest,                       # 가장 가까운 점포 예시
        "top_categories_in_radius": top_categories,  # 반경 안 업종 상위 5개 (업종 무관)
    }
