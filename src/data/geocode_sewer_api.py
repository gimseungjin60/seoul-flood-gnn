"""T4 용 센서 좌표: fetch_sewer_api 로 받은 구의 센서 주소(PSTN_INFO)를 Kakao 로 좌표로 바꾼다. 기존 geocoder.geocode(엄격 규칙: 결과 1건 + 같은 구일 때만 좌표)를 그대로 쓴다.

실행: python -m src.data.geocode_sewer_api   (.env 의 KAKAO_REST_API_KEY 필요, 지도/로컬 서비스가 켜진 앱의 REST 키)
입력: data/raw/sewer/api2025/*.csv  출력: data/processed/api2025_flood_gu_geocoding.csv
"""
import html
import os

import pandas as pd
from dotenv import load_dotenv

from src.data.clean_nodes import PROCESSED
from src.data.fetch_sewer_api import GUS, OUT_DIR
from src.data.geocoder import geocode
from src.data.inspect_water_level import PROJECT_ROOT

OUT_CSV = PROCESSED / "api2025_flood_gu_geocoding.csv"


def sensors() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f, usecols=["UNQ_NO", "SE_CD", "PSTN_INFO"], dtype=str) for f in sorted(OUT_DIR.glob("*.csv"))], ignore_index=True)
    loc = d.groupby("UNQ_NO").PSTN_INFO.agg(lambda s: s.mode().iat[0]).map(html.unescape)
    gu = d.groupby("UNQ_NO").SE_CD.first()
    return pd.DataFrame({"node_id": loc.index, "district_code": gu.loc[loc.index].to_numpy(),
                         "district_name": [GUS[str(c).zfill(2)] for c in gu.loc[loc.index]],
                         "search_keyword": loc.to_numpy(), "geocoding_status": "PENDING", "geocoding_method": "NONE",
                         "latitude": None, "longitude": None})


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("KAKAO_REST_API_KEY")
    if not key:
        raise SystemExit(".env 에 KAKAO_REST_API_KEY 가 없습니다.")
    cur = sensors()
    old = pd.read_csv(OUT_CSV, dtype={"node_id": str, "district_code": str}) if OUT_CSV.exists() else cur.iloc[0:0]
    new = cur[~cur.node_id.isin(old.node_id)]                       # 이미 변환한 센서는 다시 호출하지 않는다
    out = pd.concat([old, geocode(new, key)], ignore_index=True) if len(new) else old
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"센서 {len(out)}개 → 저장 {OUT_CSV.name}")
    print(out.groupby(["district_name", "geocoding_status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
