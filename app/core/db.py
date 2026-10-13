"""DB 연결 헬퍼. 요청마다 짧게 연결하고 닫는다 (MVP 규모에서는 커넥션 풀 없이 충분)."""
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)  # pgvector 타입을 파이썬 리스트/넘파이와 주고받을 수 있게 등록
        yield conn
