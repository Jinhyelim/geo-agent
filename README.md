# 상권·입지 분석 AI Agent (geo-agent)

자연어 질문(예: "광화문 근처에서 카페 창업하면 경쟁이 어때?")을 받아
**공간 데이터(PostGIS) 분석 + 문서 검색(RAG)** 도구를 스스로 골라 쓰고,
근거와 함께 답하는 AI Agent입니다. 분석 범위: 서울 종로구.

> 작성 중입니다.

## 실행 방법

```powershell
# 1) DB 실행 (PostgreSQL + PostGIS + pgvector)
docker compose up -d --build

# 2) 라이브러리 설치
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) 환경 변수
copy .env.example .env   # 값 채우기

# 4) 상가 데이터 적재
.\.venv\Scripts\python.exe -m scripts.load_stores data\raw\<서울 상가정보 CSV>
```
