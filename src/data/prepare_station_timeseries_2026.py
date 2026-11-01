"""강남구 2026-06~08 수위 시계열(1분) 정렬 + 센서 선정 + 품질 검증. 2025판(prepare_station_timeseries)과 같은 규칙.

실행: python -m src.data.prepare_station_timeseries_2026
입력(읽기만): data/raw/sewer/2026/DrainpipeMonitoringInfo_2026{06,07,08}.csv (cp949, 날짜 01-JUN-26 00:00:00)
출력: data/processed/gangnam2026_station_timeseries.npz     (선정 센서 values, mask, times, node_ids, is_core)
      data/processed/gangnam2026_sensor_selection.csv       (강남 전체 센서별 커버리지·역할·위치정보)
      data/processed/gangnam2026_timeseries_node_month_quality.csv, gangnam2026_timeseries_quality_report.txt

센서 선정(사전 고정, 관측 유무만 보고 수위 값·모델 결과는 보지 않는다):
- core: 06-01~08-31 1분 커버리지 >= 0.90 → 1분 연속 분석(T2 평가 대상)
- pool: 10분 블록(10분 중 5분 이상 관측, node_blocks와 같은 정의) 커버리지 >= 0.50 → 이웃 입력. core는 pool에 포함
2025판과 다른 점:
- 신규 센서는 초가 0이 아닌 시각(예 00:00:37)에 기록한다. 2025판처럼 그리드 밖으로 버리면 신규 센서가 전부 빠지므로
  분 단위로 내림해 1분 칸에 넣는다(초≠0 행 수는 보고). 같은 칸에 값이 다른 기록이 둘 이상이면 2025판과 같이 결측 처리.
- 원본 값은 수정하지 않고, 보간하지 않는다. 신규 센서 다수의 75~87분 주기 공백도 그대로 결측(mask=0)으로 둔다.
품질 규칙(원본 값 점검으로 정함, 모델 결과 없음. 해당 칸은 값을 바꾸지 않고 결측 처리만, 센서 선정은 적용 후 커버리지로):
- R1 오류 코드: 값 == -1 → 결측. 신규 센서 21개에서 -1만 음수로 나오고 23-0092는 99.8%가 -1
- R2 시스템 이상 구간: R1 후 1분 커버리지 0.90 이상 센서 중 관측 10개 이상, 그중 50% 이상이 정수값(x.00)인 분을
  표시(주기 공백 신규 센서는 이상이 없고 비율만 희석해서 판정에서 뺀다), 10분 이내 간격이면 한 구간,
  앞뒤 5분을 더해 모든 센서 결측. 2026-07-07 13:54~14:40 강우 0 mm인데 기존 계열 센서가 0·1·2 m를 번갈아 기록,
  06-09·06-25에는 전 센서가 1분간 동시에 0. 정상 시간대에 정수값 비율이 50%를 넘는 분은 없었다
"""
import time

import numpy as np
import pandas as pd

from src.data.analyze_location_match import EN_TO_KO, LOC_COL
from src.data.assign_rain_gauges import node_blocks
from src.data.clean_nodes import CODE_COL, ID_COL, PROCESSED, TIME_COL
from src.data.inspect_water_level import read_chunks
from src.data.prepare_station_timeseries import (GAP_BINS, GAP_LABELS, LEVEL_COL, STATUS_COL, gap_runs,
                                                 node_month_quality, value_checks)
from src.data.raw_paths import SEWER_2026_CSVS, SEWER_2026_TIME_FMT

GANGNAM_CODE = "23"
MONTHS = ("06", "07", "08")
START, END, FREQ = pd.Timestamp("2026-06-01 00:00"), pd.Timestamp("2026-09-01 00:00"), "1min"  # END 미포함
CORE_MIN_COVERAGE = 0.90   # 사전 고정
POOL_BLOCK_COVERAGE = 0.50  # 사전 고정
BLOCK_MIN_OBS = 5
SENTINEL = -1.0               # R1
ARTIFACT_MIN_SENSORS = 10     # R2
ARTIFACT_INT_SHARE = 0.50
ARTIFACT_MERGE_MIN = 10
ARTIFACT_PAD_MIN = 5
NODES_2025_CSV = PROCESSED / "gangnam_station_graph_nodes.csv"

OUT_NPZ = PROCESSED / "gangnam2026_station_timeseries.npz"
OUT_SELECTION = PROCESSED / "gangnam2026_sensor_selection.csv"
OUT_NODE_MONTH = PROCESSED / "gangnam2026_timeseries_node_month_quality.csv"
REPORT = PROCESSED / "gangnam2026_timeseries_quality_report.txt"


def extract_rows() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """강남(구분코드 23) 행만 청크로 추출하면서 시각·수위를 바로 변환(메모리 절약). 위치정보는 센서별로 따로 집계."""
    parts, locs = [], []
    info = {"file_rows": {}, "target_rows": {}, "bad_time": 0, "bad_level": 0, "empty_level": 0,
            "sec_nonzero": 0, "out_period": 0, "status": {}}
    for m in MONTHS:
        p = SEWER_2026_CSVS[m]
        n_total = n_target = 0
        for chunk in read_chunks(p):
            chunk = chunk.rename(columns=EN_TO_KO)
            n_total += len(chunk)
            c = chunk[chunk[CODE_COL] == GANGNAM_CODE]
            n_target += len(c)
            t = pd.to_datetime(c[TIME_COL], format=SEWER_2026_TIME_FMT, errors="coerce")
            lv = pd.to_numeric(c[LEVEL_COL], errors="coerce")
            info["bad_time"] += int(t.isna().sum())
            info["empty_level"] += int(c[LEVEL_COL].isna().sum())
            info["bad_level"] += int((c[LEVEL_COL].notna() & lv.isna()).sum())
            info["sec_nonzero"] += int((t.notna() & (t != t.dt.floor(FREQ))).sum())
            inp = (t >= START) & (t < END)
            info["out_period"] += int((t.notna() & ~inp).sum())
            for k, v in c[STATUS_COL].value_counts(dropna=False).items():
                info["status"][k] = info["status"].get(k, 0) + int(v)
            locs.append(c.groupby(ID_COL)[LOC_COL].agg(lambda s: s.dropna().value_counts().to_dict()))
            parts.append(pd.DataFrame({"node_id": c[ID_COL].to_numpy(), "time": t.dt.floor(FREQ).to_numpy(),
                                       "level": lv.to_numpy(np.float64)})[inp.to_numpy()])
            print(f"  {p.name}: {n_total:,}행, 강남 {n_target:,}", end="\r")
        print()
        info["file_rows"][m], info["target_rows"][m] = n_total, n_target
    d = pd.concat(parts, ignore_index=True)
    loc = {}
    for s in locs:
        for nid, vc in s.items():
            acc = loc.setdefault(nid, {})
            for k, v in vc.items():
                acc[k] = acc.get(k, 0) + v
    loc_tbl = pd.DataFrame([{"node_id": k, "location": max(v, key=v.get) if v else "", "n_location_variants": len(v)}
                            for k, v in loc.items()])
    return d, info, loc_tbl


def minute_cells(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(node, 1분 칸) 중복: 값이 모두 같으면 1개, 다르면 conflict(결측). 2025판 duplicate_stats와 같은 규칙."""
    keys = ["node_id", "time"]
    dupd = d.duplicated(keys, keep=False)
    single = d[~dupd].assign(n=1, conflict=False)
    g = d[dupd].groupby(keys, sort=False)["level"]
    multi = g.agg(n="size", nuniq=lambda s: s.nunique(dropna=False), level="first").reset_index()
    multi["conflict"] = multi["nuniq"] > 1
    multi.loc[multi["conflict"], "level"] = np.nan
    agg = pd.concat([single, multi.drop(columns="nuniq")], ignore_index=True)
    stats = {"dup_keys": len(multi), "dup_extra_rows": int((multi["n"] - 1).sum()),
             "conflict_keys": int(multi["conflict"].sum())}
    return agg, stats


def build_matrix(agg: pd.DataFrame, node_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(START, END, freq=FREQ, inclusive="left", name="time")
    values = agg.pivot(index="time", columns="node_id", values="level").reindex(index=index, columns=node_ids)
    mask = values.notna().astype(np.int8)
    values.columns.name = mask.columns.name = None
    return values, mask


def apply_quality_rules(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """R1(-1 오류 코드)과 R2(시스템 이상 구간)를 결측으로 바꾼 행렬과 mask, 처리 내역을 돌려준다."""
    v = values.copy()
    sentinel = v.eq(SENTINEL)
    v = v.mask(sentinel)
    ref = v.notna().mean() >= CORE_MIN_COVERAGE   # 판정은 연속 기록 센서로만(주기 공백 센서가 비율을 희석하지 않게)
    x = v.loc[:, ref].to_numpy(np.float64)
    obs = np.isfinite(x)
    isint = obs & (np.abs(x * 100 - np.round(x) * 100) < 0.5)
    n = obs.sum(axis=1)
    flag = (n >= ARTIFACT_MIN_SENSORS) & (isint.sum(axis=1) >= ARTIFACT_INT_SHARE * n)
    windows = []
    for i in np.where(flag)[0]:
        if windows and i - windows[-1][1] <= ARTIFACT_MERGE_MIN:
            windows[-1][1] = i
        else:
            windows.append([i, i])
    drop = np.zeros(len(v), bool)
    for a, b in windows:
        drop[max(a - ARTIFACT_PAD_MIN, 0):b + ARTIFACT_PAD_MIN + 1] = True
    n_window_cells = int(np.isfinite(v.to_numpy()[drop]).sum())
    v.iloc[drop] = np.nan
    mask = v.notna().astype(np.int8)
    info = {"sentinel_cells": {k: int(c) for k, c in sentinel.sum().items() if c}, "ref_sensors": int(ref.sum()),
            "flag_minutes": int(flag.sum()), "window_cells": n_window_cells,
            "windows": [(v.index[max(a - ARTIFACT_PAD_MIN, 0)], v.index[min(b + ARTIFACT_PAD_MIN, len(v) - 1)], int(flag[a:b + 1].sum()))
                        for a, b in windows]}
    return v, mask, info


def select_sensors(values: pd.DataFrame, mask: pd.DataFrame, loc_tbl: pd.DataFrame) -> pd.DataFrame:
    blk = node_blocks(values.to_numpy(), mask.to_numpy(), min_obs=BLOCK_MIN_OBS)
    old18 = set(pd.read_csv(NODES_2025_CSV)["node_id"])
    sel = pd.DataFrame({"node_id": values.columns, "minute_coverage": mask.mean().to_numpy(),
                        "block_coverage": np.isfinite(blk).mean(axis=0)})
    sel["is_core"] = sel["minute_coverage"] >= CORE_MIN_COVERAGE
    sel["in_pool"] = sel["is_core"] | (sel["block_coverage"] >= POOL_BLOCK_COVERAGE)
    sel["role"] = np.where(sel["is_core"], "core", np.where(sel["in_pool"], "neighbor", "excluded"))
    sel["in_2025_18"] = sel["node_id"].isin(old18)
    return sel.merge(loc_tbl, on="node_id", how="left")


def cycle_stats(mask: pd.DataFrame) -> pd.DataFrame:
    """결측 구간 길이 중앙값으로 주기 공백(75~87분)형 센서를 구분하기 위한 요약."""
    rows = []
    for node in mask.columns:
        runs = gap_runs(mask[node].to_numpy())
        rows.append({"node_id": node, "n_gaps": len(runs), "gap_p50_min": float(np.median(runs)) if len(runs) else 0.0,
                     "gaps_per_day": len(runs) / (len(mask) / 1440)})
    return pd.DataFrame(rows)


def build_report(info, dup, agg, n_obs, qc, values, mask, sel, cyc, nm_pool, vchk, elapsed) -> str:
    L = []
    w = L.append
    w("[강남 2026] 수위 시계열 정렬·센서 선정·품질 리포트 (모델 학습 없음)")
    w("=" * 66)
    w("입력: " + ", ".join(f"data/raw/sewer/2026/{SEWER_2026_CSVS[m].name}" for m in MONTHS))
    w(f"시간축: {START} ~ {END - pd.Timedelta(FREQ)} ({FREQ}, {len(mask):,}행), 강남 센서 {mask.shape[1]}개")
    w("시간대: 원본에 표기 없음 → 변환하지 않음(현지 시각 추정)")
    w(f"처리 시간 {elapsed:,.0f}초")

    w("\n1. 레코드 수 대조")
    for m in MONTHS:
        w(f"  2026-{m}: 원본 {info['file_rows'][m]:,}행 중 강남 {info['target_rows'][m]:,}행")
    tot = sum(info["target_rows"].values())
    n_nan_keys = int((agg["level"].isna() & ~agg["conflict"]).sum())
    rhs = info["bad_time"] + info["out_period"] + dup["dup_extra_rows"] + dup["conflict_keys"] + n_nan_keys + n_obs
    w(f"  강남 행 {tot:,} = 시각 파싱 실패 {info['bad_time']:,} + 기간 밖 {info['out_period']:,} "
      f"+ 같은 칸 초과행 {dup['dup_extra_rows']:,} + 값 충돌 칸 {dup['conflict_keys']:,} "
      f"+ 수위 빈값/비숫자 칸 {n_nan_keys:,} + 관측 칸 {n_obs:,} → 설명 안 된 차이 {tot - rhs:,}")
    w(f"  초≠0 행 {info['sec_nonzero']:,} ({info['sec_nonzero'] / max(tot, 1):.1%}) → 분 단위 내림으로 1분 칸에 배정")
    w(f"  수위 빈값 {info['empty_level']:,}행, 비숫자 {info['bad_level']:,}행 / 통신상태 {info['status']}")

    w("\n1-1. 품질 규칙 (값은 바꾸지 않고 결측 처리만)")
    s = qc["sentinel_cells"]
    w(f"  R1 값 == {SENTINEL:g} → 결측: {sum(s.values()):,}칸, 센서 {len(s)}개: "
      + ", ".join(f"{k} {v:,}" for k, v in sorted(s.items(), key=lambda kv: -kv[1])))
    w(f"  R2 시스템 이상 구간(R1 후 1분 커버리지 ≥ {CORE_MIN_COVERAGE:.2f} 센서 {qc['ref_sensors']}개 기준, "
      f"관측 {ARTIFACT_MIN_SENSORS}개 이상 중 정수값 {ARTIFACT_INT_SHARE:.0%} 이상인 분, "
      f"{ARTIFACT_MERGE_MIN}분 이내 병합, 앞뒤 {ARTIFACT_PAD_MIN}분): 표시 분 {qc['flag_minutes']}개, "
      f"결측 처리 {qc['window_cells']:,}칸")
    for a, b, k in qc["windows"]:
        w(f"    {a:%m-%d %H:%M} ~ {b:%H:%M} (표시 분 {k}개)")
    w(f"  관측 칸: 규칙 적용 전 {n_obs:,} → 적용 후 {int(mask.values.sum()):,}")

    w("\n2. 센서 선정 (사전 고정: core 1분 커버리지 ≥ "
      f"{CORE_MIN_COVERAGE:.2f}, pool 10분 블록 커버리지 ≥ {POOL_BLOCK_COVERAGE:.2f}, 블록 유효 = 10분 중 {BLOCK_MIN_OBS}분 이상)")
    for role in ("core", "neighbor", "excluded"):
        s = sel[sel.role == role]
        w(f"  {role}: {len(s)}개 (2025 18개 중 {int(s.in_2025_18.sum())}개)")
    w(f"  pool 합계(core+neighbor): {int(sel.in_pool.sum())}개")
    w("  core: " + ", ".join(f"{r.node_id}({r.minute_coverage:.2f}{'*' if r.in_2025_18 else ''})"
                             for r in sel[sel.is_core].itertuples()) + "   (*=2025 18개)")
    w("  2025 18개 중 core가 아닌 센서: " + (", ".join(f"{r.node_id}(분 {r.minute_coverage:.2f}, 블록 {r.block_coverage:.2f}, {r.role})"
                                                for r in sel[sel.in_2025_18 & ~sel.is_core].itertuples()) or "없음"))
    w("  커버리지 분포(분 단위): " + ", ".join(f"q{int(q * 100)} {sel.minute_coverage.quantile(q):.2f}"
                                       for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    w("  커버리지 분포(10분 블록): " + ", ".join(f"q{int(q * 100)} {sel.block_coverage.quantile(q):.2f}"
                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    w(f"  위치정보 문구가 둘 이상인 센서: {int((sel.n_location_variants > 1).sum())}개, "
      f"위치정보 없음: {int((sel.n_location_variants.fillna(0) == 0).sum())}개")

    w("\n3. 결측 구조 (보간하지 않음)")
    c = cyc.merge(sel[["node_id", "role"]], on="node_id")
    for role, g in c.groupby("role"):
        w(f"  {role}: 하루 결측 구간 수 중앙값 {g.gaps_per_day.median():.1f}, 결측 구간 길이 중앙값의 중앙값 "
          f"{g.gap_p50_min.median():.0f}분")
    pool = sel[sel.in_pool].node_id.tolist()
    runs = np.concatenate([gap_runs(mask[n].to_numpy()) for n in pool])
    hist = pd.cut(pd.Series(runs), GAP_BINS, right=False, labels=GAP_LABELS).value_counts().reindex(GAP_LABELS)
    w("  pool 연속 결측 길이 분포: " + ", ".join(f"{k} {v}" for k, v in hist.items()))
    w("  pool 월별 결측률(센서 평균 / 최대):")
    for m, g in nm_pool.groupby("month"):
        w(f"    {m}: {g.missing_rate.mean():.2%} / {g.missing_rate.max():.2%} ({g.loc[g.missing_rate.idxmax(), 'node_id']})")

    w("\n4. 값 점검 (pool, 값은 수정하지 않음)")
    w(vchk.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    w(f"\n  마스크 검증: NaN ↔ mask=0 일치 {bool((values.isna().values == (mask.values == 0)).all())}")
    w("\n5. 다음 단계")
    w("  - pool 센서 좌표 확보(위치정보 → 지오코딩). 2025에 좌표가 없던 23-0011, 23-0018 재시도")
    w("  - 2026 재비교 판정 규칙(첨두 시각 오차 포함)을 결과 확인 전에 코드에 고정")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.perf_counter()
    d, info, loc_tbl = extract_rows()
    agg, dup = minute_cells(d)
    del d
    node_ids = sorted(agg["node_id"].unique())
    raw_values, raw_mask = build_matrix(agg, node_ids)
    assert not raw_values.index.duplicated().any()
    values, mask, qc = apply_quality_rules(raw_values)
    n_obs_raw = int(raw_mask.values.sum())
    del raw_values, raw_mask
    sel = select_sensors(values, mask, loc_tbl)
    cyc = cycle_stats(mask)
    pool = sel[sel.in_pool].node_id.tolist()
    nm_pool = node_month_quality(mask[pool])
    vchk = value_checks(values[pool])

    is_core = sel.set_index("node_id").loc[pool, "is_core"].to_numpy()
    np.savez_compressed(OUT_NPZ, values=values[pool].to_numpy(np.float32), mask=mask[pool].to_numpy(np.int8),
                        times=values.index.to_numpy().astype("datetime64[s]"),
                        node_ids=np.array(pool, dtype=str), is_core=is_core)
    with np.load(OUT_NPZ) as z:  # allow_pickle 없이 로드되는지 확인
        assert z["values"].shape == (len(mask), len(pool)) and list(z["node_ids"]) == pool
    sel.merge(cyc, on="node_id").to_csv(OUT_SELECTION, index=False, encoding="utf-8-sig", float_format="%.4f")
    nm_pool.to_csv(OUT_NODE_MONTH, index=False, encoding="utf-8-sig")
    report = build_report(info, dup, agg, n_obs_raw, qc, values, mask, sel, cyc, nm_pool, vchk, time.perf_counter() - t0)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
