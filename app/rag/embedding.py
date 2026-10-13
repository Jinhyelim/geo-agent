"""OpenAI 임베딩 호출. 문서 적재와 질문 검색이 같은 모델을 써야 벡터를 비교할 수 있다."""
from openai import OpenAI

from app.core.config import settings

_client = OpenAI(api_key=settings.openai_api_key, timeout=20, max_retries=2)


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(model=settings.openai_embedding_model, input=texts)
    return [d.embedding for d in resp.data]
