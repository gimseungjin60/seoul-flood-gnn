"""서울 열린데이터광장 OpenAPI(DrainpipeMonitoringInfo)로 2025년 침수 피해가 난 구의 하루 단위 1분 수위를 받는다(침수흔적도와 맞추기 위한 T4 자료).
인증키는 .env 의 SEOUL_OPENAPI_KEY(git 제외). 원본 CSV 를 쓰지 않는 이유: 서울 전체 월 파일이 2 GB 라 필요한 구·날짜만 API 로 받는다.

실행: python -m src.data.fetch_sewer_api            # 기본 구 5개 x 2025-08-10~08-25
      python -m src.data.fetch_sewer_api --gu 12 --days 20250813
출력: data/raw/sewer/api2025/{구코드}_{YYYYMMDD}.csv (이미 있으면 건너뜀, git 제외). 열: UNQ_NO, SE_CD, SE_NM, MSRMT_YMD, MSRMT_WATL, SGN_STTS, PSTN_INFO.
구분코드 SE_CD: 응답의 SE_NM 으로 확인한다(09 강북, 10 도봉, 12 은평, 13 서대문, 16 강서, 23 강남).
"""
import argparse
import json
import time
import urllib.request

import pandas as pd

from src.dashboard.live import api_key
from src.data.raw_paths import SEWER_API_2025

OUT_DIR = SEWER_API_2025
BASE = "http://openapi.seoul.go.kr:8088"
PAGE = 1000
GUS = {"12": "은평", "16": "강서", "13": "서대문", "10": "도봉", "09": "강북"}
COLS = ["UNQ_NO", "SE_CD", "SE_NM", "MSRMT_YMD", "MSRMT_WATL", "SGN_STTS", "PSTN_INFO"]
DAYS = [f"202508{d:02d}" for d in range(10, 26)]


def fetch(key: str, gu: str, day: str) -> pd.DataFrame:
    rows, start, total = [], 1, None
    while total is None or start <= total:
        url = f"{BASE}/{key}/json/DrainpipeMonitoringInfo/{start}/{start + PAGE - 1}/{gu}/{day}00/{day}23/"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    raw = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:                                       # 키가 든 주소를 노출하지 않는다
                if attempt == 2:
                    raise RuntimeError(f"{gu} {day} 호출 실패: {type(e).__name__}") from None
                time.sleep(2)
        if "DrainpipeMonitoringInfo" not in raw:                         # 자료가 없는 날은 서비스 키 없이 RESULT 만 온다
            res = raw.get("RESULT", {})
            if res.get("CODE") == "INFO-200":
                return pd.DataFrame(columns=COLS)
            raise RuntimeError(f"{gu} {day}: {res.get('CODE')} {res.get('MESSAGE', '')}")
        body = raw["DrainpipeMonitoringInfo"]
        if body["RESULT"]["CODE"] != "INFO-000":
            raise RuntimeError(f"{gu} {day}: {body['RESULT']['CODE']} {body['RESULT'].get('MESSAGE', '')}")
        total = body["list_total_count"]
        rows += body.get("row", [])
        start += PAGE
    return pd.DataFrame(rows, columns=COLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gu", nargs="*", default=list(GUS))
    ap.add_argument("--days", nargs="*", default=DAYS)
    args = ap.parse_args()
    key = api_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for gu in args.gu:
        for day in args.days:
            out = OUT_DIR / f"{gu}_{day}.csv"
            if out.exists():
                continue
            t0 = time.time()
            d = fetch(key, gu, day)
            names = sorted(d.SE_NM.unique()) if len(d) else []
            d.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"{gu} {day}: {len(d):,}행, 센서 {d.UNQ_NO.nunique() if len(d) else 0}개, SE_NM={names} ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
