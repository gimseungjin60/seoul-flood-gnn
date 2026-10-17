"""서울시 강우량 정보(10분우량, 2025-05~08) 정렬 + 품질 검증 + 강우 이벤트 목록 + 수위 반응 확인 (원본은 읽기만 한다).

실행: python -m src.data.prepare_rainfall
입력: data/raw/서울시_강우량_정보_2025년{05,06,07,08}월.csv (cp949), gangnam_station_timeseries.npz
출력: data/processed/rainfall_10min_2025.npz        48개 강우량계 x 10분 격자(구간 끝 시각 라벨)
      data/processed/gangnam_rain_events.csv        강남 3개소 평균 기준 강우 이벤트
      data/processed/rainfall_quality_report.txt

값의 의미: 자료수집시각 13:09 = 13:00:00~13:09:59 10분 누적 강우량(mm). 격자 라벨은 구간 끝(:09, :19, ...)이며
           '예측 시점 t의 입력'으로 쓰려면 라벨 시각 <= t 인 구간만 써야 한다(구간이 끝난 뒤에만 알 수 있음).
결측: 파일에 행이 없는 칸은 NaN(0mm로 채우지 않는다). mask=1이 원본에 있는 값.
이벤트(사전 고정): 강남 3개소 평균 10분우량 > 0 인 구간을 6시간 이상 무강우로 끊어 묶는다. 총량 10 mm 이상이면 '유의 이벤트'.
"""
import time

import numpy as np
import pandas as pd

from src.data.clean_nodes import PROCESSED
from src.data.inspect_water_level import PROJECT_ROOT
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ

RAW = PROJECT_ROOT / "data" / "raw"
MONTHS = ("05", "06", "07", "08")
FILES = [RAW / f"서울시_강우량_정보_2025년{m}월.csv" for m in MONTHS]
OUT_NPZ = PROCESSED / "rainfall_10min_2025.npz"
OUT_EVENTS = PROCESSED / "gangnam_rain_events.csv"
REPORT = PROCESSED / "rainfall_quality_report.txt"
GANGNAM = "강남구"
DRY_GAP_STEPS = 36          # 6시간 / 10분
SIGNIFICANT_MM = 10.0
TRAIN_END, VAL_END = 92_736, 112_608   # 수위 분 인덱스(2025-06-01 00:00 기준), 70/15/15 분할


def read_raw(files=FILES) -> pd.DataFrame:
    parts = [pd.read_csv(f, encoding="cp949", dtype=str) for f in files]
    d = pd.concat(parts, ignore_index=True)
    d.columns = ["gid", "gname", "gu_code", "gu", "r10", "ts"][: d.shape[1]]
    # '00:29:59'처럼 초가 붙은 행, '10:18'처럼 수집 시각이 몇 분 어긋난 행(약 0.03%)이 있다.
    # 수집 시각이 속한 10분 구간의 끝 라벨(:09,:19,...)로 매핑한다: 내림(10분) + 9분. 정상 행과 충돌하지 않음을 확인했다.
    d["t_raw"] = pd.to_datetime(d["ts"], format="mixed").dt.floor("min")
    d["t"] = d["t_raw"].dt.floor("10min") + pd.Timedelta(minutes=9)
    d["r"] = pd.to_numeric(d["r10"], errors="raise")
    return d


def to_grid(d: pd.DataFrame):
    """long → (T, G) 격자(라벨=구간 끝). 같은 (지점, 구간)에 행이 둘이면 에러(합산하면 이중 집계가 되므로)."""
    if d.duplicated(["gid", "t"]).any():
        raise ValueError(f"같은 지점·10분 구간에 중복 행 {int(d.duplicated(['gid', 't']).sum())}개")
    t0, t1 = d["t"].min(), d["t"].max()
    times = pd.date_range(t0, t1, freq="10min")
    info = d.drop_duplicates("gid").set_index("gid")[["gname", "gu"]].sort_index()
    gids = list(info.index)
    piv = d.pivot_table(index="t", columns="gid", values="r", aggfunc="sum")  # (gid,t) 중복 없음 확인은 리포트에서
    piv = piv.reindex(index=times, columns=gids)
    values = piv.to_numpy(dtype=np.float32)
    return times, gids, info, values


def find_events(rain: np.ndarray, times: pd.DatetimeIndex, dry_gap: int = DRY_GAP_STEPS) -> pd.DataFrame:
    """rain (T,) mm/10분(NaN 허용). >0 구간을 dry_gap 스텝 이상 무강우(또는 결측)로 끊어 이벤트로 묶는다."""
    wet = np.nan_to_num(rain, nan=0.0) > 0
    idx = np.flatnonzero(wet)
    rows, start = [], None
    for k, i in enumerate(idx):
        if start is None:
            start, last = i, i
        elif i - last > dry_gap:
            rows.append((start, last))
            start, last = i, i
        else:
            last = i
    if start is not None:
        rows.append((start, last))
    out = []
    for s, e in rows:
        seg = np.nan_to_num(rain[s:e + 1], nan=0.0)
        out.append({"start": times[s] - pd.Timedelta(minutes=9), "end": times[e], "total_mm": float(seg.sum()),
                    "max_10min_mm": float(seg.max()), "peak_time_end": times[s + int(seg.argmax())],
                    "duration_h": (e - s + 1) / 6})
    return pd.DataFrame(out)


def network_anomaly(s_times: pd.DatetimeIndex, s_vals: np.ndarray, s_mask: np.ndarray, train_end: int = TRAIN_END) -> pd.Series:
    """전 노드 anomaly(10분 평균 - 노드 train 중앙값)의 노드 간 중앙값. 인덱스=10분 구간 끝 라벨(:09...). 강우 격자와 같은 구간."""
    v = np.where(s_mask > 0, s_vals, np.nan).astype(np.float64)
    med = np.nanmedian(v[:train_end], axis=0)
    t = len(v) // 10 * 10
    x = v[:t].reshape(-1, 10, v.shape[1])
    cnt = np.isfinite(x).sum(axis=1)
    blk = np.where(cnt >= 5, np.where(np.isfinite(x), x, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    with np.errstate(all="ignore"):
        net = np.nanmedian(blk - med, axis=1)
    return pd.Series(net, index=s_times[:t][9::10])


def sewer_response(ev: pd.DataFrame, net: pd.Series) -> pd.DataFrame:
    """이벤트마다 전 노드 anomaly 중앙값의 최대 시각·크기(이벤트 구간 + 이후 3시간).
    강우 피크 대비 지연은 계산하지 않는다: 이벤트 하나에 호우가 여러 번 있으면(예: 08-13~14) 지연이 무의미해진다."""
    rows = []
    for e in ev.itertuples():
        w = net[(net.index >= e.start) & (net.index <= e.end + pd.Timedelta(hours=3))]
        ok = w.dropna()
        rows.append({"start": e.start, "sewer_peak_time": ok.idxmax() if len(ok) else pd.NaT,
                     "sewer_peak_m": float(ok.max()) if len(ok) else np.nan, "sewer_obs_share": float(w.notna().mean())})
    return pd.DataFrame(rows)


def lag_corr(rain: pd.Series, net: pd.Series, max_lag: int = 18) -> pd.DataFrame:
    """강우(10분) vs 수위 반응(10분 anomaly의 변화량)의 교차상관. lag k = 강우 구간 뒤 k칸(10분) 후 수위 변화."""
    idx = rain.index.intersection(net.index)
    r, y = rain.reindex(idx), net.reindex(idx).diff()
    rows = []
    for k in range(0, max_lag + 1):
        a, b = r.iloc[:len(r) - k].to_numpy(), y.iloc[k:].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        rows.append({"lag_min": k * 10, "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 200 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    d = read_raw()
    dup = int(d.duplicated(["gid", "t"]).sum())
    off = (d["t_raw"].dt.minute % 10) != 9
    times, gids, info, values = to_grid(d)
    mask = np.isfinite(values).astype(np.int8)
    np.savez_compressed(OUT_NPZ, times=times.to_numpy().astype("datetime64[s]"), gauge_ids=np.array(gids),
                        gauge_names=info["gname"].to_numpy(), gauge_gu=info["gu"].to_numpy(), values=values, mask=mask)

    gn = [i for i, g in enumerate(gids) if info.loc[g, "gu"] == GANGNAM]
    names = [info.loc[gids[i], "gname"] for i in gn]
    with np.errstate(all="ignore"):
        mean_gn = np.nanmean(values[:, gn], axis=1)
    sew = np.load(SEWER_NPZ, allow_pickle=True)
    s_times = pd.DatetimeIndex(sew["times"])
    win = (times >= "2025-06-01") & (times < "2025-09-01")
    ev = find_events(np.where(win, mean_gn, np.nan), times)
    sig = ev[ev.total_mm >= SIGNIFICANT_MM].reset_index(drop=True)
    net = network_anomaly(s_times, sew["values"], sew["mask"])
    resp = sewer_response(sig, net)
    lc = lag_corr(pd.Series(np.where(win, mean_gn, np.nan), index=times), net)
    sig = sig.merge(resp, on="start")
    sig.insert(0, "event_id", range(1, len(sig) + 1))

    def split_of(ts):
        m = int((ts - s_times[0]).total_seconds() // 60)
        return "train" if m < TRAIN_END else ("val" if m < VAL_END else "test")
    sig["split_by_start"] = [split_of(s) for s in sig["start"]]
    sig["split_by_end"] = [split_of(s) for s in sig["end"]]
    sig.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig", float_format="%.3f")

    L = []
    w = L.append
    w("[서울시 강우량 정보 2025-05~08] 정렬·품질·이벤트 리포트 (원본 수정 없음)")
    w("=" * 70)
    w(f"입력 {len(FILES)}개 파일, {len(d):,}행, 강우량계 {len(gids)}개, 중복(gid,시각) {dup}행, 10분 격자 {len(times):,}칸 "
      f"({times[0]} ~ {times[-1]}, 라벨=구간 끝)")
    w(f"[사실] 수집 시각이 :09 규칙을 벗어난 행 {int(off.sum())}개(전체의 {off.mean():.3%}, 그중 강우>0 {int((d.r[off] > 0).sum())}개): "
      "10분 구간 끝 라벨로 내림 매핑. 같은 구간에 정상 행이 없어 충돌·이중 집계 없음. 초(:59) 포함 행도 같은 규칙.")
    w(f"[사실] 값은 0.5 mm 단위(전체 고유값 {len(np.unique(values[np.isfinite(values)]))}개), 음수 {int((values < 0).sum())}개, 최대 {np.nanmax(values):.1f} mm/10분")
    w("")
    w("1. 결측 (격자 대비 행 없음 → NaN, 0으로 채우지 않음)")
    miss = (mask == 0).sum(axis=0)
    w(f"  전체 결측 칸 {int((mask == 0).sum()):,} / {mask.size:,} ({(mask == 0).mean():.3%}), 지점별 결측 min/median/max = "
      f"{miss.min()}/{int(np.median(miss))}/{miss.max()}")
    allmiss = np.flatnonzero((mask == 0).all(axis=1))
    w(f"  전 지점 동시 결측 시각 {len(allmiss)}개" + (f": {times[allmiss][:5].tolist()} ..." if len(allmiss) else ""))
    w(f"  강남구 3개소 {names} 결측: " + ", ".join(f"{names[k]} {int(miss[i])}" for k, i in enumerate(gn)))
    w("")
    w(f"2. 수위 분석 기간(2025-06-01~08-31) 강남 3개소 (mm)")
    for k, i in enumerate(gn):
        x = np.where(win, values[:, i], np.nan)
        w(f"  {names[k]}: 총 {np.nansum(x):.1f}, 최대 10분 {np.nanmax(x):.1f}")
    cm = pd.DataFrame(values[win][:, gn], columns=names).corr().round(3)
    w("  지점 간 10분우량 상관:\n" + cm.to_string())
    near = []
    for g in ("서초구", "송파구", "동작구", "용산구"):
        cols = [i for i, k in enumerate(gids) if info.loc[k, "gu"] == g]
        with np.errstate(all="ignore"):
            other = np.nanmean(values[:, cols], axis=1)
        near.append(f"{g} 평균과의 상관 " + f"{pd.Series(mean_gn[win]).corr(pd.Series(other[win])):.3f}")
    w("  강남 평균 vs 인접 구 평균: " + ", ".join(near))
    w("")
    w(f"3. 강우 이벤트 (강남 3개소 평균, 6시간 무강우로 분리, 총량 {SIGNIFICANT_MM:.0f} mm 이상) — {len(sig)}개 / 전체 {len(ev)}개")
    cols = ["event_id", "start", "end", "total_mm", "max_10min_mm", "split_by_start", "split_by_end", "sewer_peak_m",
            "sewer_obs_share"]
    w(sig[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    w(f"  분할별 유의 이벤트(시작 기준): {sig.split_by_start.value_counts().to_dict()}, (끝 기준): {sig.split_by_end.value_counts().to_dict()}")
    w("  sewer_peak_m = 이벤트 구간+3시간 안에서 전 노드 anomaly 중앙값(10분 평균, 노드 train 중앙값 기준)의 최대(m).")
    w("  이벤트 하나에 호우가 여러 번 있을 수 있다(예: 08-13~14는 08-13 새벽과 08-14 새벽 두 번). 강우 피크~수위 피크 지연은 이 때문에 이벤트별로 계산하지 않는다.")
    w("")
    w("  강우(강남 평균, 10분) → 수위 반응(전 노드 anomaly 중앙값의 10분 변화량) 교차상관, 분석 기간 전체")
    best = lc.loc[lc["corr"].idxmax()]
    w("   " + ", ".join(f"{int(r.lag_min)}분 {r.corr:.3f}" for r in lc.itertuples() if r.lag_min in (0, 10, 20, 30, 60, 120, 180)))
    w(f"   최대 상관 시차 {int(best.lag_min)}분 (상관 {best['corr']:.3f})")
    w("")
    w("4. 수위 결측과의 겹침")
    for a, b in (("2025-06-17 15:55", "2025-06-17 17:17"), ("2025-06-21 23:30", "2025-06-21 23:33")):
        m = (times >= pd.Timestamp(a).floor("10min")) & (times <= pd.Timestamp(b).ceil("10min") + pd.Timedelta(minutes=9))
        w(f"  수위 전 노드 동시 결측 {a}~{b[-5:]}: 같은 시간 강남 평균 강우 합 {np.nansum(mean_gn[m]):.1f} mm")
    w("")
    w("주의")
    w("  - 강우량계 위치(위경도) 파일은 아직 없다. 지금은 '강남구 3개소 평균'을 강남 강우로 쓴다. 센서별 최근접 강우량계 배정에는 위치 파일이 필요하다.")
    w("  - 이벤트 수는 강남 평균 기준이며 센서마다 맞은 비는 다를 수 있다.")
    w("  - 2025-05월은 선행 강수 계산용으로만 쓰고, 이벤트 목록은 수위 분석 기간만 대상이다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
