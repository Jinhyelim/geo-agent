"""문서 검색 평가: 벡터만 / 하이브리드(완화 없음) / 하이브리드(현재) 비교.

사용법:
    python -m eval.run_eval

지표 (top_k=4, 실제 서비스와 같은 값)
  - Hit@4     : 정답 쪽이 하나라도 결과 4개 안에 있으면 성공 (문서에 있는 질문 20개 기준)
  - Recall@4  : 정답 쪽 중 몇 %를 찾았나 (질문별 평균)
  - 거절률     : 문서에 없는 질문(5개)에 "근거 없음"으로 답한 비율 (높을수록 좋음)

질문과 정답 쪽은 청크 원문을 직접 보고 지정했다 (소규모 세트라 수치는 방향을 보는 용도).
"""
import json
from pathlib import Path

from app.rag import retriever as R

TOP_K = 4
DATA = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
DOCS = DATA["docs"]


def found_pages(result: dict) -> set[tuple[str, int]]:
    if not result.get("found"):
        return set()
    pages = set()
    for h in result["results"]:
        for key, name in DOCS.items():
            if name in h["source"]:
                pages.add((key, h["page"]))
    return pages


def evaluate(mode: str, relax: bool, scope: bool) -> dict:
    original, original_scope = R.KEYWORD_MIN_SIMILARITY, R.OUT_OF_SCOPE_CHECK
    R.KEYWORD_MIN_SIMILARITY = original if relax else R.MIN_SIMILARITY  # 완화 없음 = 0.4 그대로
    R.OUT_OF_SCOPE_CHECK = scope
    rows = []
    try:
        for item in DATA["questions"]:
            res = R.search_docs(item["q"], top_k=TOP_K, mode=mode)
            got = found_pages(res)
            gold = {(d, p) for d, p in item["relevant"]}
            rows.append({"id": item["id"], "type": item["type"], "gold": gold, "got": got,
                         "found": bool(res.get("found"))})
    finally:
        R.KEYWORD_MIN_SIMILARITY, R.OUT_OF_SCOPE_CHECK = original, original_scope
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    return {
        "rows": rows,
        "hit": sum(bool(r["gold"] & r["got"]) for r in pos) / len(pos),
        "recall": sum(len(r["gold"] & r["got"]) / len(r["gold"]) for r in pos) / len(pos),
        "reject": sum(not r["found"] for r in neg) / len(neg),
        "pos": len(pos), "neg": len(neg),
    }


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


if __name__ == "__main__":
    settings = [("① 벡터만 (이전 방식)", "vector", False, False),
                ("② 하이브리드", "hybrid", False, False),
                ("③ ② + 고유명사 완화", "hybrid", True, False),
                ("④ ③ + 범위 밖 지명 거절 (현재)", "hybrid", True, True)]
    results = [(name, evaluate(mode, relax, scope)) for name, mode, relax, scope in settings]

    lines = [f"# 문서 검색 평가 결과 (top_k={TOP_K})", "",
             f"문서에 있는 질문 {results[0][1]['pos']}개, 문서에 없는 질문 {results[0][1]['neg']}개", "",
             "| 방식 | Hit@4 | Recall@4 | 문서에 없는 질문 거절률 |", "|---|---|---|---|"]
    for name, r in results:
        lines.append(f"| {name} | {pct(r['hit'])} | {pct(r['recall'])} | {pct(r['reject'])} |")

    lines += ["", "## 질문 유형별 Hit@4", "", "| 유형 | " + " | ".join(n for n, _ in results) + " |",
              "|---|" + "---|" * len(results)]
    for t in ["설명", "지명", "표"]:
        cells = []
        for _, r in results:
            sub = [x for x in r["rows"] if x["type"] == t]
            cells.append(f"{sum(bool(x['gold'] & x['got']) for x in sub)}/{len(sub)}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", "## 질문별 결과 (찾은 정답 쪽 / 전체 정답 쪽)", "",
              "| id | 질문 | " + " | ".join(n for n, _ in results) + " |", "|---|---|" + "---|" * len(results)]
    for i, item in enumerate(DATA["questions"]):
        cells = []
        for _, r in results:
            row = r["rows"][i]
            if row["gold"]:
                cells.append(f"{len(row['gold'] & row['got'])}/{len(row['gold'])}")
            else:
                cells.append("거절 ✅" if not row["found"] else "근거 반환 ❌")
        lines.append(f"| {item['id']} | {item['q']} | " + " | ".join(cells) + " |")

    out = "\n".join(lines)
    (Path(__file__).parent / "results.md").write_text(out + "\n", encoding="utf-8")
    print(out)
