"""소상공인시장진흥공단 상가(상권)정보 CSV에서 종로구 점포만 골라 stores 테이블에 적재한다.

사용법:
    python -m scripts.load_stores data/raw/<내려받은 서울 CSV 파일>.csv
"""
import sys
from pathlib import Path

import pandas as pd
import psycopg

from app.core.config import settings

TARGET_SIGUNGU = "종로구"

# CSV 한글 컬럼명 -> DB 컬럼명
COLUMNS = {
    "상가업소번호": "store_id",
    "상호명": "name",
    "상권업종대분류명": "category_large",
    "상권업종중분류명": "category_mid",
    "상권업종소분류명": "category_small",
    "시군구명": "sigungu",
    "행정동명": "dong",
    "도로명주소": "road_address",
    "경도": "lon",
    "위도": "lat",
}


def read_csv(path: Path) -> pd.DataFrame:
    # 공공데이터 CSV는 배포 시기에 따라 UTF-8 또는 CP949(EUC-KR)로 저장되어 있다.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 판별할 수 없습니다: {path}")


def main(csv_path: str) -> None:
    df = read_csv(Path(csv_path))

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}\n실제 컬럼: {list(df.columns)}")

    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df = df[df["sigungu"] == TARGET_SIGUNGU]

    # 좌표가 없거나 숫자가 아닌 행은 공간 질의에 쓸 수 없으므로 제외한다.
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["store_id", "name", "lon", "lat"])
    print(f"[load] {TARGET_SIGUNGU} {before}건 중 좌표가 유효한 {len(df)}건 적재")

    rows = [
        (r.store_id, r.name, r.category_large, r.category_mid, r.category_small,
         r.sigungu, r.dong, r.road_address, r.lon, r.lat)
        for r in df.itertuples(index=False)
    ]

    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        # 다시 실행해도 결과가 같도록(멱등) 같은 점포 번호는 덮어쓴다.
        cur.executemany(
            """
            INSERT INTO stores (store_id, name, category_large, category_mid, category_small,
                                sigungu, dong, road_address, geom)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)
            ON CONFLICT (store_id) DO UPDATE SET
                name = EXCLUDED.name,
                category_large = EXCLUDED.category_large,
                category_mid = EXCLUDED.category_mid,
                category_small = EXCLUDED.category_small,
                dong = EXCLUDED.dong,
                road_address = EXCLUDED.road_address,
                geom = EXCLUDED.geom
            """,
            rows,
        )
        cur.execute("SELECT count(*) FROM stores")
        print(f"[load] stores 테이블 총 {cur.fetchone()[0]}건")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
