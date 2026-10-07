"""Kakao Local API 지오코딩 (주소 검색 → 결과 없으면 키워드 검색). 이번 단계에서는 실행하지 않는다.

실행(키 설정 후): python -m src.data.geocoder
입력: data/processed/gangnam_geocoding_input.csv (변경하지 않음)
출력: data/processed/gangnam_geocoding_result.csv

좌표를 추측하지 않는다: 결과가 정확히 1건이고 그 주소가 대상 구 안일 때만 위도/경도를 쓴다.
  OK            1건, 대상 구 안 → latitude/longitude 기록
  MULTIPLE      2건 이상 → 좌표 비움, 후보 수만 기록(수동 검토)
  OUT_OF_DISTRICT 1건이지만 다른 구 → 좌표 비움
  NOT_FOUND     주소·키워드 검색 모두 0건
  ERROR         HTTP/네트워크 오류
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
from dotenv import load_dotenv

from src.data.clean_nodes import PROCESSED
from src.data.inspect_water_level import PROJECT_ROOT

IN_CSV = PROCESSED / "gangnam_geocoding_input.csv"
OUT_CSV = PROCESSED / "gangnam_geocoding_result.csv"
ENDPOINTS = {"ADDRESS": "https://dapi.kakao.com/v2/local/search/address.json",
             "KEYWORD": "https://dapi.kakao.com/v2/local/search/keyword.json"}
REQUEST_INTERVAL_S = 0.2  # ponytail: 고정 간격, 대량 호출 시 429 재시도 로직 추가


def kakao_search(query: str, method: str, api_key: str) -> list[dict]:
    url = f"{ENDPOINTS[method]}?{urllib.parse.urlencode({'query': query})}"
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {api_key}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["documents"]


def decide(docs: list[dict], district_full: str) -> tuple[str, float | None, float | None, str | None]:
    """검색 결과 → (status, lat, lon, matched_address). 1건 + 대상 구일 때만 좌표."""
    if not docs:
        return "NOT_FOUND", None, None, None
    if len(docs) > 1:
        return "MULTIPLE", None, None, None
    d = docs[0]
    addr = d.get("address_name") or ""
    if district_full not in addr.split():
        return "OUT_OF_DISTRICT", None, None, addr
    return "OK", float(d["y"]), float(d["x"]), addr  # Kakao: x = 경도, y = 위도


def geocode(df: pd.DataFrame, api_key: str, search=kakao_search) -> pd.DataFrame:
    out = df.copy()
    for col in ("n_results", "matched_address"):
        out[col] = None
    for i, r in out[out["geocoding_status"] == "PENDING"].iterrows():
        district_full = f"{r['district_name']}구"
        try:
            for method in ("ADDRESS", "KEYWORD"):
                docs = search(r["search_keyword"], method, api_key)
                time.sleep(REQUEST_INTERVAL_S)
                if docs:
                    break
            status, lat, lon, addr = decide(docs, district_full)
            out.loc[i, ["geocoding_status", "geocoding_method", "latitude", "longitude",
                        "n_results", "matched_address"]] = [status, method, lat, lon, len(docs), addr]
        except Exception as e:  # 개별 노드 실패가 전체를 멈추지 않게
            out.loc[i, ["geocoding_status", "matched_address"]] = ["ERROR", repr(e)[:200]]
    return out


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        sys.exit(".env에 KAKAO_REST_API_KEY가 없습니다. .env.example을 복사해 설정하세요.")
    df = pd.read_csv(IN_CSV, dtype={"district_code": str})
    out = geocode(df, api_key)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(out["geocoding_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
