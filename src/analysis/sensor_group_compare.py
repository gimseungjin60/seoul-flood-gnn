"""2025-08 강남 센서: 기존 20개(HIGH)와 신규 99개(LOW)가 같은 종류인지 비교한다 (진단 전 점검, 읽기 전용).

비교 창: 2025-08-05 00:00 ~ 08-17 23:59 (공간 진단 후보 기간). 08-18 이후는 test 구간이라 읽자마자 버리고 요약에도 쓰지 않는다.
비교 항목: 창 안 기록 수·커버리지·측정 간격, 평소 수위 분포, 변동 폭, 큰 이벤트 상승폭, 주소 패턴.

실행: python -m src.analysis.sensor_group_compare   (원본 2.2 GB를 읽어 약 2~4분)
출력: data/processed/gangnam_202508_new_vs_old_node_stats.csv, gangnam_202508_new_vs_old_report.txt

결과(2025-08-05~08-17): 신규 99개 중 1,000건 이상 기록한 센서가 0개(중앙값 5건), 평소 수위 중앙값이 기존 0.125 대비 2.635로 달라서
이 창에서는 신규 센서를 쓸 수 없다. 자세한 해석은 docs/09_midterm_qa.md 참고.
"""
import re
import sys

import numpy as np
import pandas as pd

from src.data.analyze_location_match import EN_TO_KO, LOC_COL
from src.data.clean_nodes import CODE_COL, ID_COL, PROCESSED, TIME_COL
from src.data.inspect_water_level import read_chunks
from src.data.raw_paths import SEWER_2025_CSVS

WINDOW_START = pd.Timestamp("2025-08-05 00:00:00")
WINDOW_END = pd.Timestamp("2025-08-18 00:00:00")  # 배타적: 08-17 23:59까지
EXPECTED_ROWS = 13 * 1440
GANGNAM_CODE = "23"
QUALITY_CSV = PROCESSED / "gangnam_202508_quality.csv"
EVENTS_CSV = PROCESSED / "gangnam_rain_events.csv"
OUT_STATS = PROCESSED / "gangnam_202508_new_vs_old_node_stats.csv"
OUT_REPORT = PROCESSED / "gangnam_202508_new_vs_old_report.txt"
LEVEL_COL = "측정수위"


def address_class(text: str) -> str:
    """주소 문구의 도로명 유형(간선/지선 구분이 아니라 문구 형태 분류): 대로 / 로+길·길 / 로 / 도로명 없음."""
    t = str(text)
    if "대로" in t:
        return "대로"
    if re.search(r"[가-힣0-9]+로\s*\d+번?길|[가-힣0-9]+길", t):
        return "로+길/길"
    if re.search(r"[가-힣0-9]+로", t):
        return "로"
    return "도로명 없음"


def is_jibun_only(text: str) -> bool:
    """도로명(로/길) 없이 '동 번지'로 끝나는 주소."""
    t = str(text)
    return bool(re.search(r"[가-힣]+동\s*\d+(-\d+)?\s*$", t)) and not re.search(r"로|길", t)


def load_window(path=SEWER_2025_CSVS[2]) -> pd.DataFrame:
    """강남 행 중 비교 창 안의 행만 읽는다 (창 밖은 즉시 버림)."""
    parts = []
    for ch in read_chunks(path):
        ch = ch.rename(columns=EN_TO_KO)
        ch = ch[ch[CODE_COL] == GANGNAM_CODE]
        t = pd.to_datetime(ch[TIME_COL], errors="coerce")
        keep = (t >= WINDOW_START) & (t < WINDOW_END)
        d = ch.loc[keep, [ID_COL, LEVEL_COL, LOC_COL]].copy()
        d["t"] = t[keep]
        parts.append(d)
    g = pd.concat(parts, ignore_index=True)
    g["v"] = pd.to_numeric(g[LEVEL_COL], errors="coerce")
    return g


def node_stats(d: pd.DataFrame) -> pd.Series:
    d = d.sort_values("t")
    dt = d["t"].drop_duplicates().diff().dt.total_seconds().dropna()
    v = d["v"].dropna()
    return pd.Series({
        "n": len(d), "coverage": d["t"].nunique() / EXPECTED_ROWS,
        "interval_med_s": dt.median(), "gap_p95_min": dt.quantile(0.95) / 60,
        "mean": v.mean(), "median": v.median(), "std": v.std(), "p05": v.quantile(0.05), "p95": v.quantile(0.95),
        "max": v.max(), "min": v.min(), "zero_frac": (v == 0).mean(),
    })


def dry_mask(times: pd.Series, events: pd.DataFrame) -> pd.Series:
    """이벤트 구간(시작-1h ~ 끝+3h) 밖이면 True (무강우 시간대)."""
    m = pd.Series(True, index=times.index)
    for _, r in events.iterrows():
        m &= ~((times >= r["start"] - pd.Timedelta(hours=1)) & (times <= r["end"] + pd.Timedelta(hours=3)))
    return m


def main() -> None:
    q = pd.read_csv(QUALITY_CSV)
    old = set(q.loc[q.quality_grade == "HIGH", "node_id"])
    g = load_window()
    g["grp"] = np.where(g[ID_COL].isin(old), "기존20", "신규99")
    st = g.groupby(["grp", ID_COL]).apply(node_stats, include_groups=False).reset_index()
    st.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")

    ev = pd.read_csv(EVENTS_CSV, parse_dates=["start", "end"])
    ev = ev[(ev["end"] >= WINDOW_START) & (ev["start"] < WINDOW_END)]
    mask = dry_mask(g["t"], ev)
    base = g[mask].groupby(ID_COL)["v"].median()
    big = ev.sort_values("total_mm").iloc[-1]
    seg = g[(g["t"] >= big["start"] - pd.Timedelta(hours=1)) & (g["t"] <= big["end"] + pd.Timedelta(hours=3))]
    rise = (seg.groupby(ID_COL)["v"].max() - base).dropna()
    grp_of = dict(zip(g[ID_COL], g["grp"]))
    rise_df = pd.DataFrame({"rise": rise, "grp": rise.index.map(grp_of)})

    loc = g.groupby(ID_COL)[LOC_COL].agg(lambda x: x.value_counts().index[0]).to_frame("loc")
    loc["grp"] = loc.index.map(grp_of)
    loc["cls"] = loc["loc"].map(address_class)
    loc["jibun_only"] = loc["loc"].map(is_jibun_only)
    loc["manhole"] = loc["loc"].str.contains("맨홀")

    pd.set_option("display.width", 200)
    cols = ["n", "coverage", "interval_med_s", "gap_p95_min", "mean", "median", "std", "p95", "max", "zero_frac"]
    out = [f"비교 창: {WINDOW_START:%Y-%m-%d} ~ {WINDOW_END - pd.Timedelta(minutes=1):%Y-%m-%d %H:%M} (13일, 센서당 최대 {EXPECTED_ROWS:,}건). 이후 구간은 읽지 않음.",
           f"센서 수(창 안 기록 있음): {st.groupby('grp')[ID_COL].nunique().to_dict()} (신규 99개 중 나머지는 창 안 기록 없음)", "",
           "[사실] 센서별 지표의 그룹별 중앙값", st.groupby("grp")[cols].median().round(3).to_string(), "",
           "[사실] 신규 센서 창 안 기록 수 분포: " + str(st.loc[st.grp == "신규99", "n"].describe().round(1).to_dict()),
           "[사실] 기록 1,000건 이상: " + str(st.assign(f=st.n >= 1000).groupby("grp")["f"].sum().to_dict()),
           "[사실] 측정 간격(중앙값) 90초 초과: " + str(st.assign(f=st.interval_med_s > 90).groupby("grp")["f"].sum().to_dict()),
           "[사실] 센서별 평균 수위 1 이상: " + str(st.assign(f=st["mean"] >= 1).groupby("grp")["f"].sum().to_dict()), "",
           "창과 겹치는 이벤트 (7·8번 2개뿐, 둘 다 val 구간):", ev[["event_id", "start", "end", "total_mm", "max_10min_mm", "split_by_start"]].to_string(index=False), "",
           f"[사실] 큰 이벤트({int(big['event_id'])}번, {big['total_mm']:.0f} mm) 상승폭(구간 최대 - 무강우 기저) 그룹별:", rise_df.groupby("grp")["rise"].describe().round(3).to_string(),
           "(신규 센서는 기록이 5건 안팎이라 이 상승폭에 의미를 두기 어렵다)", "",
           "[사실] 주소 패턴 (도로명 유형, 문구 형태일 뿐 간선/지선 구분 근거는 아님):", pd.crosstab(loc.grp, loc.cls).to_string(),
           "동+번지만 있는 주소 / '맨홀' 포함 비율:", loc.groupby("grp")[["jibun_only", "manhole"]].mean().round(2).to_string(), "",
           "[해석] 신규 99개는 이 창 안에서 시계열이 없다(기록 중앙값 5건). 평소 수위도 센서 평균 중앙값 2.6 대 0.125로 크게 달라",
           "       설치·시험 단계 기록일 가능성이 있다(추정, 확인 필요). 2026-09 자료에서는 신규 센서의 평균 수위가 기존과 비슷했다.",
           "[결정] 이 창에서는 신규 센서를 쓸 수 없다 → 지오코딩·진단은 하지 않는다."]
    OUT_REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
