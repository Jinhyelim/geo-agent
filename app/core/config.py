"""환경 변수(.env)를 읽어 설정 객체로 제공한다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    vworld_api_key: str = ""
    kakao_rest_api_key: str = ""

    database_url: str = "postgresql://geo:geo@localhost:5433/geo"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # 외부 API 호출 정책 (보내드림 프로젝트의 timeout/재시도 원칙을 그대로 가져옴)
    http_timeout_seconds: float = 3.0
    http_max_attempts: int = 3


settings = Settings()
