"""대시보드 실시간 모드: 서울 열린데이터광장 OpenAPI에서 강남구 최신 수위·강우를 읽는다. 인증키는 .env 의 SEOUL_OPENAPI_KEY (git 제외).

- 수위: DrainpipeMonitoringInfo/{시작}/{끝}/{구분코드 23=강남}/{YYYYMMDDHH}/{YYYYMMDDHH} (OA-2527 요청인자). 수위 단위 m. 약 10분 지연.
- 강우: ListRainfallService/{시작}/{끝}. 10분 강우(RN_10M, mm), 최신이 앞쪽. 강남구 강우량계만 쓴다.
- 평소 대비 상승폭 = (수위 − 그 센서의 요일x시 중앙값[2026-06~08]) x 1000 mm. **평소 기준이 여름 자료라 계절이 다르면 달라질 수 있다.**
- 시각은 서울(KST) 현지 시각으로 맞춘다. 호출 실패·키 없음은 예외 대신 LiveError 로 알려 화면이 이유를 보여 준다.
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from src.analysis.spatial_signal_profile_adjusted import hour_of_week

BASE = "http://openapi.seoul.go.kr:8088"
GANGNAM_CD = "23"
PAGE = 1000
KST = timezone(timedelta(hours=9))
ERROR_LEVEL_M = -0.99   # 원자료의 오류 코드 -1 (data/README.md)
STALE_MIN = 90          # 이 시간보다 오래된 마지막 값은 '자료 없음'으로 본다(신규 센서는 약 2시간 주기로 기록)


class LiveError(RuntimeError):
    pass


def now_kst() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(KST).replace(tzinfo=None))


def api_key() -> str:
    key = (dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("SEOUL_OPENAPI_KEY") or "").strip()
    if not key:
        raise LiveError(".env 에 SEOUL_OPENAPI_KEY 가 없습니다(열린데이터광장 인증키).")
    return key


def _get(service: str, *args) -> dict:
    url = "/".join([BASE, api_key(), "json", service, *map(str, args)])
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:                                          # 네트워크·JSON 오류는 키가 들어간 주소를 노출하지 않고 알린다
        raise LiveError(f"{service} 호출 실패: {type(e).__name__}") from None
    body = d.get(service)
    res = (body or d).get("RESULT", {})
    if res.get("CODE") != "INFO-000":
        raise LiveError(f"{service} 응답 오류: {res.get('CODE')} {res.get('MESSAGE', '')}".strip())
    return body


def _pages(service: str, *tail, limit: int) -> list[dict]:
    rows, start = [], 1
    while start <= limit:
        end = min(start + PAGE - 1, limit)
        body = _get(service, start, end, *tail)
        rows += body.get("row", [])
        if end >= body["list_total_count"]:
            break
        start = end + 1
    return rows


def fetch_sewer(now: pd.Timestamp, hours: int = 3) -> pd.DataFrame:
    """강남구 센서의 최근 `hours`시간 수위. 열: node_id, time, level_m, status."""
    s, e = (now - pd.Timedelta(hours=hours)).strftime("%Y%m%d%H"), now.strftime("%Y%m%d%H")
    rows = _pages("DrainpipeMonitoringInfo", GANGNAM_CD, s, e, limit=20000)
    if not rows:
        raise LiveError("수위 자료가 비어 있습니다.")
    d = pd.DataFrame(rows)
    return pd.DataFrame({"node_id": d.UNQ_NO, "time": pd.to_datetime(d.MSRMT_YMD).dt.floor("s"),
                         "level_m": pd.to_numeric(d.MSRMT_WATL, errors="coerce"), "status": d.SGN_STTS}).query("level_m > @ERROR_LEVEL_M")


def fetch_rain() -> pd.DataFrame:
    """강남구 강우량계의 최근 10분 강우. 열: time, gauge, mm. 최신 1000행(전체 강우량계 약 2시간치)에서 강남구만 고른다."""
    rows = _get("ListRainfallService", 1, PAGE).get("row", [])
    d = pd.DataFrame([r for r in rows if r.get("GU_NM") == "강남구"])
    if d.empty:
        raise LiveError("강남구 강우 자료가 없습니다.")
    return pd.DataFrame({"time": pd.to_datetime(d.DATA_CLCT_TM), "gauge": d.RF_NM,
                         "mm": pd.to_numeric(d.RN_10M, errors="coerce")}).sort_values("time")


def latest_rise(sewer: pd.DataFrame, nodes: list[str], prof: np.ndarray, now: pd.Timestamp) -> pd.DataFrame:
    """센서별 가장 최근 값과 평소 대비 상승폭(mm). 열: node_id, time, age_min, level_m, rise_mm, status, ok.
    ok = 통신양호이고 STALE_MIN 안에 측정된 값만 True. prof 는 (168, 센서) 요일x시 중앙값(m)이고 nodes 순서와 같다."""
    last = sewer.sort_values("time").groupby("node_id").tail(1).set_index("node_id")
    out = []
    for j, n in enumerate(nodes):
        if n not in last.index or not np.isfinite(last.at[n, "level_m"]):
            out.append(dict(node_id=n, time=pd.NaT, age_min=np.nan, level_m=np.nan, rise_mm=np.nan, status="자료 없음", ok=False))
            continue
        t, lv = last.at[n, "time"], float(last.at[n, "level_m"])
        how = int(hour_of_week(pd.DatetimeIndex([t]))[0])
        age = (now - t).total_seconds() / 60
        st = last.at[n, "status"]
        out.append(dict(node_id=n, time=t, age_min=age, level_m=lv, rise_mm=(lv - prof[how, j]) * 1000, status=st,
                        ok=bool(st == "통신양호" and 0 <= age <= STALE_MIN)))
    return pd.DataFrame(out)


def rise_series(sewer: pd.DataFrame, node: str, j: int, prof: np.ndarray) -> pd.DataFrame:
    """한 센서의 최근 수위를 평소 대비 상승폭(mm)으로. 열: time, rise_mm."""
    s = sewer[(sewer.node_id == node) & np.isfinite(sewer.level_m)].sort_values("time")
    if s.empty:
        return pd.DataFrame({"time": [], "rise_mm": []})
    how = hour_of_week(pd.DatetimeIndex(s.time))
    return pd.DataFrame({"time": s.time.to_numpy(), "rise_mm": (s.level_m.to_numpy() - prof[how, j]) * 1000})
