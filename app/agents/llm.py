"""LLM(OpenAI) 호출 공통 설정. timeout과 재시도를 한곳에서 관리한다."""
from openai import OpenAI

from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key, timeout=30, max_retries=2)
MODEL = settings.openai_chat_model
