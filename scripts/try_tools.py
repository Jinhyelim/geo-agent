"""도구를 하나씩 단독으로 실행해 보는 스크립트.

사용법:
    python -m scripts.try_tools
"""
import json

from app.tools.geocode import geocode
from app.tools.spatial import count_stores_nearby


def show(title: str, data: dict) -> None:
    print(f"\n=== {title}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    for q in ["광화문", "종로3가역", "세종대로 172", "사직동", "없는장소xyz"]:
        show(f"geocode('{q}')", geocode(q))

    g = geocode("광화문")
    if g["ok"]:
        show("광화문 반경 500m 카페", count_stores_nearby(g["lat"], g["lon"], 500, "카페"))
