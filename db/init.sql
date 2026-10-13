-- 컨테이너 최초 기동 시 1회 실행된다.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- 상가(상권)정보: 한 행 = 한 점포
CREATE TABLE IF NOT EXISTS stores (
    store_id       TEXT PRIMARY KEY,           -- 상가업소번호
    name           TEXT NOT NULL,              -- 상호명
    category_large TEXT,                       -- 상권업종대분류명 (예: 음식)
    category_mid   TEXT,                       -- 상권업종중분류명 (예: 비알코올)
    category_small TEXT,                       -- 상권업종소분류명 (예: 카페)
    sigungu        TEXT,                       -- 시군구명
    dong           TEXT,                       -- 행정동명
    road_address   TEXT,                       -- 도로명주소
    -- geography 타입: 위경도(SRID 4326)를 지구 곡률을 고려해 '미터' 단위로 거리 계산할 수 있다.
    geom           GEOGRAPHY(POINT, 4326) NOT NULL
);

-- 반경 검색(ST_DWithin)이 전체 테이블을 훑지 않도록 공간 인덱스(GIST)를 건다.
CREATE INDEX IF NOT EXISTS idx_stores_geom ON stores USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_stores_category ON stores (category_small);
CREATE INDEX IF NOT EXISTS idx_stores_dong ON stores (dong);

-- RAG 문서 청크
CREATE TABLE IF NOT EXISTS doc_chunks (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT NOT NULL,        -- 문서 파일명
    page       INT,                  -- 페이지 번호 (출처 표시용)
    content    TEXT NOT NULL,
    embedding  VECTOR(1536) NOT NULL -- text-embedding-3-small 차원
);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON doc_chunks USING hnsw (embedding vector_cosine_ops);
