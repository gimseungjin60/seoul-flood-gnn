"""T5c: 2026년 9월(학습·검증에 쓰지 않은 새 달)에 T5 모델을 적용해 보는 참고용 점검. 규칙은 결과를 보기 전에 고정했다(2026-11-09). **판정 없음, 기술 통계만.**

[왜 판정이 없는가] 9월은 비가 거의 없던 달이다. 강남 평균 강우로 총 10 mm 이상인 사건이 1개(2026-09-01, 23 mm, 10분 최대 1.2 mm)뿐이라 사건 단위 부트스트랩이 불가능하다.
  그래서 신뢰구간이나 '낫다/구분 불가' 판정을 하지 않고, 한 사건의 센서별 오차를 있는 그대로 보고한다.
[모델] T5 와 같다(src/analysis/t5_peak.py: 센서별 효과 릿지 회귀 하나, 기준 B0·B1·B2, 기점 = 사건 시작 2시간 후). 학습 = 2025 사건 + 2026(06~08) 사건 전부(t5b 의 B 평가와 같은 풀링).
  센서 = 2025·2026 공통 17개. 평소 수위 기준 = 2026-06~08 수위의 요일x시 중앙값(여름 기준, 9월 값으로 새로 만들지 않음 → 인과적).
[9월 자료] 수위: data/raw/sewer/2026/DrainpipeMonitoringInfo_202609.csv 에서 강남(se_cd 23) 센서만, 10분 블록 평균(오류값 -1(<= -0.99) 제외). 강우: 서울시_강우량_정보_2026년09월.csv, 강남 강우량계 평균.
  사건 = prepare_rainfall.find_events 규칙(총 10 mm 이상, 6시간 무강우 분리). 사건 창 = 시작 ~ 종료 + 3시간.
[보고] 사건의 센서별 실제 최대 상승, 모델·기준 예측, 사건 평균 MAE, 모델 오차가 B2 보다 작은 센서 수. 9월 수위의 월 전체 기록 안정성(센서 수 등).
[한계] 사건 1개, 약한 비. 이 점검은 어떤 판정도 바꾸지 않는다.
실행: python -m src.analysis.t5c_sept      출력: data/processed/gangnam_t5c_sept_report.txt
"""
import numpy as np
import pandas as pd

from src.analysis.rise_levels import rise_mm, stage
from src.analysis.spatial_signal_profile_adjusted import hour_of_week
from src.analysis.t5_peak import MODELS, ORIGIN_MAIN, predict
from src.analysis.t5b_peak import pairs as pairs_generic
from src.analysis.t2_loso import build_inputs as inputs_2025
from src.analysis.t5_peak import gn_rain_2025
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import GANGNAM, SIGNIFICANT_MM, find_events, read_raw, to_grid
from src.data.raw_paths import RAIN_2026_09, SEWER_2026_CSVS

OUT_REPORT = PROCESSED / "gangnam_t5c_sept_report.txt"
ERR_M = -0.99
CHUNK = 1_000_000


def sept_levels(ids) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(SEWER_2026_CSVS["09"], encoding="cp949", usecols=["unq_no", "se_cd", "msrmt_ymd", "msrmt_watl"], dtype=str, chunksize=CHUNK):
        ch = ch[ch.unq_no.isin(ids)]
        if len(ch):
            parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d["t"] = pd.to_datetime(d.msrmt_ymd, format="%d-%b-%y %H:%M:%S").dt.floor("10min")
    d["lv"] = pd.to_numeric(d.msrmt_watl, errors="coerce")
    d.loc[~(d.lv > ERR_M), "lv"] = np.nan
    p = d.pivot_table(index="t", columns="unq_no", values="lv", aggfunc="mean")
    return p.reindex(pd.date_range(p.index.min().normalize(), p.index.max().normalize() + pd.Timedelta(hours=23, minutes=50), freq="10min"))


def main() -> None:
    from src.dashboard.data import load
    D = load()
    i25 = inputs_2025()
    ids25, ids26 = i25["nodes"], D["nodes"]
    stable = set(D["info"].node_id[~D["info"].unstable])
    ids = [n for n in ids25 if n in set(ids26) and n in stable]
    # 학습: T5b 의 풀링(2025 + 2026 여름)
    r25 = rise_mm({"blk": i25["blk"], "how": i25["how"]})[:, [ids25.index(n) for n in ids]]
    r26 = D["rise"][:, [ids26.index(n) for n in ids]]
    train = pd.concat([pairs_generic(i25["bt"], i25["events"], r25, gn_rain_2025(i25["bt"]), ids, 2025),
                       pairs_generic(D["bt"], D["events"], r26, D["rain_gn"], ids, 2026)], ignore_index=True)
    # 9월
    lv = sept_levels(ids)
    prof = D["prof"][:, [ids26.index(n) for n in ids]]                      # 2026-06~08 요일x시 중앙값(m)
    how = hour_of_week(lv.index)
    cols = [n for n in ids if n in lv.columns]
    rise = np.full((len(lv), len(ids)), np.nan)
    for j, n in enumerate(ids):
        if n in lv.columns:
            rise[:, j] = (lv[n].to_numpy() - prof[how, j]) * 1000
    d = read_raw([RAIN_2026_09])
    times, gids, info, values = to_grid(d)
    gn = [i for i, g in enumerate(gids) if info.loc[g, "gu"] == GANGNAM]
    rt = pd.DatetimeIndex(times)
    bt = lv.index + pd.Timedelta(minutes=9)                                 # 블록 끝 라벨(:09)
    pos = rt.get_indexer(bt)
    rain = np.where(pos >= 0, np.nanmean(values[np.clip(pos, 0, None)][:, gn], axis=1), np.nan)
    ev = find_events(np.nanmean(values[:, gn], axis=1), times)
    ev = ev[ev.total_mm >= SIGNIFICANT_MM].reset_index(drop=True)
    ev.insert(0, "event_id", range(1, len(ev) + 1))
    L = ["[강남] T5c 2026-09 점검 (참고, 판정 없음, 규칙은 결과 보기 전 고정: src/analysis/t5c_sept.py)", "=" * 80,
         f"센서 {len(ids)}개(2025·2026 공통). 학습 = 2025 + 2026 여름 사건 {train.event.nunique()}개({len(train)}쌍). 9월 수위가 있는 센서 {len(cols)}개. "
         f"9월 총 10 mm 이상 사건 {len(ev)}개: " + ", ".join(f"{r.start:%m-%d %H:%M}~{r.end:%m-%d %H:%M} 총 {r.total_mm:.0f} mm, 10분 최대 {r.max_10min_mm:.1f} mm" for r in ev.itertuples()), ""]
    test = pairs_generic(bt, ev, rise, rain, ids, 2026)
    if test.empty:
        L.append("평가 가능한 (사건, 센서)가 없다.")
    else:
        te = predict(train, test, ids)
        err = {m: (te[m] - te.peak).abs() for m in MODELS}
        L.append(f"평가 (사건, 센서) {len(te)}쌍. 사건 평균 MAE(mm): " + ", ".join(f"{m} {err[m].mean():.1f}" for m in MODELS))
        L.append(f"  [진단, 규칙 변경 아님] 기점(시작 2시간 후) 누적 강우 R = {te.R.iloc[0]:.1f} mm (학습 사건의 R 중앙 {train.groupby('event').R.first().median():.1f} mm, 최소 {train.groupby('event').R.first().min():.1f} mm), "
                 f"기점 상승 a 중앙 {te.a.median():.0f} mm, 그때까지 최대 m 중앙 {te.m.median():.0f} mm")
        L.append(f"  M 의 오차가 B2 보다 작은 센서 {int((err['M'] < err['B2']).sum())}개, 같음 {int((err['M'] == err['B2']).sum())}개, 큼 {int((err['M'] > err['B2']).sum())}개 / {len(te)}개")
        L.append("  실제 최대 상승 중앙/최대: {:.0f} / {:.0f} mm (단계: ".format(te.peak.median(), te.peak.max())
                 + ", ".join(f"{n} {int((stage(te.peak.to_numpy()) == k).sum())}" for k, n in enumerate(["평상", "상승", "큰 상승", "매우 큰 상승"])) + ")")
        L += ["", "  센서별 (실제 최대 / B1 / B2 / M, mm):"]
        for r in te.sort_values("peak", ascending=False).itertuples():
            L.append(f"    {r.node}: {r.peak:6.0f} / {r.B1:6.0f} / {r.B2:6.0f} / {r.M:6.0f}")
    L += ["", "※ 사건 1개, 약한 비. 신뢰구간·판정 없음. 평소 수위 기준은 여름(06~08)이다."]
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
