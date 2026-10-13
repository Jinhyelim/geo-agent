"""PDF 문서를 잘라(청킹) 임베딩한 뒤 doc_chunks 테이블에 저장한다.

사용법:
    python -m app.rag.ingest            # data/docs 폴더의 모든 PDF
"""
from pathlib import Path

from pypdf import PdfReader

from app.core.db import get_conn
from app.rag.embedding import embed_texts

DOCS_DIR = Path("data/docs")
CHUNK_SIZE = 800     # 한 조각 글자 수. 너무 길면 검색이 뭉뚱그려지고, 너무 짧으면 문맥이 끊긴다.
CHUNK_OVERLAP = 150  # 조각 사이 겹치는 글자 수. 문장이 경계에서 잘려 의미가 사라지는 것을 줄인다.
BATCH = 64           # 임베딩 API에 한 번에 보내는 조각 수


def split_text(text: str) -> list[str]:
    text = " ".join(text.split())  # 줄바꿈·연속 공백 정리
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]  # 의미 없는 짧은 조각 제외


def load_pdf(path: Path) -> list[tuple[str, int, str]]:
    """(파일명, 페이지 번호, 조각) 목록. 페이지 번호는 답변의 '출처' 표시에 쓴다."""
    rows = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        for chunk in split_text(page.extract_text() or ""):
            rows.append((path.name, page_no, chunk))
    return rows


def main() -> None:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[ingest] {DOCS_DIR}에 PDF가 없습니다.")
        return

    with get_conn() as conn, conn.cursor() as cur:
        for pdf in pdfs:
            rows = load_pdf(pdf)
            if not rows:
                print(f"[ingest] {pdf.name}: 텍스트 추출 실패 (스캔 이미지 PDF일 수 있음) → 건너뜀")
                continue
            # 같은 문서를 다시 넣으면 중복되지 않게 먼저 지운다 (멱등)
            cur.execute("DELETE FROM doc_chunks WHERE source = %s", (pdf.name,))
            for i in range(0, len(rows), BATCH):
                batch = rows[i:i + BATCH]
                vectors = embed_texts([r[2] for r in batch])
                cur.executemany(
                    "INSERT INTO doc_chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(s, p, c, v) for (s, p, c), v in zip(batch, vectors)],
                )
            print(f"[ingest] {pdf.name}: {len(rows)}개 조각 저장")
        cur.execute("SELECT count(*) FROM doc_chunks")
        print(f"[ingest] doc_chunks 총 {cur.fetchone()[0]}개")


if __name__ == "__main__":
    main()
