"""그래프 노드 18개의 2025-06~08 수위 시계열을 1분 시간축 × 노드 idx 순서 행렬로 정렬 + 품질 검증.

실행: python -m src.data.prepare_station_timeseries
입력(읽기만): data/raw/tv_swm_wal_mea_2025{06,07,08}.csv, data/processed/gangnam_station_graph_nodes.csv
출력: data/processed/gangnam_station_timeseries.csv      (행=시각, 열=node_id(idx 순), 값=원본 측정값, 결측은 빈칸)
      data/processed/gangnam_station_observation_mask.csv (1=원본 관측, 0=결측)
      data/processed/gangnam_station_timeseries.npz       (values, mask, times, node_ids — 같은 내용, 빠른 로딩용)
      data/processed/gangnam_timeseries_node_month_quality.csv, gangnam_timeseries_quality_report.txt

- 원본 값은 수정/삭제하지 않는다. 결측은 NaN + mask=0 으로만 표시하고 이번 단계에서는 보간하지 않는다.
- 같은 노드·같은 시각에 값이 다른 중복이 있으면 어느 값도 고르지 않고 결측(mask=0)으로 두고 보고한다.
- 1분 그리드에 맞지 않는 시각(초가 0이 아님 등)은 반올림하지 않고 행렬에서 제외, 개수를 보고한다.
"""
import time

import numpy as np
import pandas as pd

from src.data.analyze_location_match import EN_TO_KO, PERIOD_CSVS
from src.data.clean_nodes import ID_COL, PROCESSED, TIME_COL, normalize_text
from src.data.inspect_water_level import read_chunks

NODES_CSV = PROCESSED / "gangnam_station_graph_nodes.csv"
OUT_VALUES = PROCESSED / "gangnam_station_timeseries.csv"
OUT_MASK = PROCESSED / "gangnam_station_observation_mask.csv"
OUT_NPZ = PROCESSED / "gangnam_station_timeseries.npz"
OUT_NODE_MONTH = PROCESSED / "gangnam_timeseries_node_month_quality.csv"
REPORT = PROCESSED / "gangnam_timeseries_quality_report.txt"
LEVEL_COL, STATUS_COL = "측정수위", "통신상태"
TIME_FMT = "%Y-%m-%d %H:%M:%S"
START, END, FREQ = pd.Timestamp("2025-06-01 00:00"), pd.Timestamp("2025-09-01 00:00"), "1min"  # END 미포함
# 사전 고정 이상값 기준(값은 바꾸지 않고 개수만 보고)
JUMP_M = 1.0          # 연속 1분 사이 |Δ수위| >= 1.0 m
GAP_BINS = [1, 2, 6, 31, 181, 1441, np.inf]  # 연속 결측 길이(분) 구간: 1, 2–5, 6–30, 31–180, 181–1440, >1440
GAP_LABELS = ["1분", "2–5분", "6–30분", "31분–3시간", "3–24시간", "24시간 초과"]


def extract_rows(paths, node_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    """원본 CSV에서 대상 노드 행만 청크 단위로 추출. ID는 정확 일치만 사용하고,
    공백 등을 정리하면 대상과 같아지는 '근접 ID'는 따로 세어 보고한다(수정해서 쓰지 않음)."""
    targets = set(node_ids)
    parts, info = [], {"file_rows": {}, "near_miss_ids": {}, "target_rows": {}}
    for p in paths:
        month = p.stem[-6:]
        n_total = n_target = 0
        for chunk in read_chunks(p):
            chunk = chunk.rename(columns=EN_TO_KO)
            n_total += len(chunk)
            ids = chunk[ID_COL]
            hit = ids.isin(targets)
            miss = ~hit & ids.map(normalize_text).isin(targets)
            for raw, n in ids[miss].value_counts().items():
                info["near_miss_ids"][repr(raw)] = info["near_miss_ids"].get(repr(raw), 0) + int(n)
            c = chunk.loc[hit, [ID_COL, TIME_COL, LEVEL_COL, STATUS_COL]]
            n_target += len(c)
            parts.append(c.assign(src_month=month))
            print(f"  {p.name}: {n_total:,}행, 대상 {n_target:,}", end="\r")
        print()
        info["file_rows"][month], info["target_rows"][month] = n_total, n_target
    df = pd.concat(parts, ignore_index=True).rename(columns={ID_COL: "node_id"})
    df["node_id"] = df["node_id"].astype("category")
    return df, info


def parse_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["time"] = pd.to_datetime(d[TIME_COL], format=TIME_FMT, errors="coerce")
    d["level"] = pd.to_numeric(d[LEVEL_COL], errors="coerce")
    d["bad_time"] = d["time"].isna()
    d["bad_level"] = d[LEVEL_COL].notna() & d["level"].isna()
    d["in_period"] = (d["time"] >= START) & (d["time"] < END)
    d["on_grid"] = d["time"].notna() & (d["time"] == d["time"].dt.floor(FREQ))
    return d


def duplicate_stats(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(node, time) 중복: 값이 같으면 1개로 합침, 값이 다르면 conflict(행렬에서 결측 처리)."""
    g = d.dropna(subset=["time"]).groupby(["node_id", "time"], observed=True)["level"]
    agg = g.agg(n="size", nuniq=lambda s: s.nunique(dropna=False), level="first").reset_index()
    stats = {"dup_keys": int((agg["n"] > 1).sum()), "dup_extra_rows": int((agg["n"] - 1).sum()),
             "conflict_keys": int((agg["nuniq"] > 1).sum())}
    agg.loc[agg["nuniq"] > 1, "level"] = np.nan
    agg["conflict"] = agg["nuniq"] > 1
    return agg, stats


def interval_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for node, g in d.dropna(subset=["time"]).groupby("node_id", observed=True):
        gaps = g["time"].drop_duplicates().sort_values().diff().dt.total_seconds().dropna()
        vc = gaps.value_counts()
        rows.append({"node_id": node, "interval_mode_s": vc.index[0], "share_60s": float((gaps == 60).mean()),
                     "n_intervals": len(gaps), "n_not_60s": int((gaps != 60).sum()),
                     "other_intervals_top": ", ".join(f"{k:g}s×{v}" for k, v in vc.iloc[1:4].items())})
    return pd.DataFrame(rows)


def build_matrix(agg: pd.DataFrame, node_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1분 그리드 × node_ids(그래프 idx 순). 그리드 밖/기간 밖 시각은 넣지 않는다."""
    index = pd.date_range(START, END, freq=FREQ, inclusive="left", name="time")
    a = agg[agg["time"].isin(index)]
    values = a.pivot(index="time", columns="node_id", values="level").reindex(index=index, columns=node_ids)
    observed = a.assign(o=a["level"].notna()).pivot(index="time", columns="node_id", values="o")
    mask = observed.reindex(index=index, columns=node_ids).fillna(False).astype(np.int8)
    values.columns.name = mask.columns.name = None
    return values, mask


def gap_runs(observed: np.ndarray) -> np.ndarray:
    """0/1 관측 배열 → 연속 결측(0) 구간 길이 배열."""
    x = np.concatenate([[1], observed.astype(np.int8), [1]])
    d = np.diff(x)
    starts, ends = np.where(d == -1)[0], np.where(d == 1)[0]
    return ends - starts


def node_month_quality(mask: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, mm in mask.groupby(mask.index.to_period("M")):
        for node in mask.columns:
            runs = gap_runs(mm[node].to_numpy())
            rows.append({"node_id": node, "month": str(month), "expected": len(mm), "observed": int(mm[node].sum()),
                         "missing": int(len(mm) - mm[node].sum()), "missing_rate": round(1 - mm[node].mean(), 6),
                         "n_gaps": len(runs), "max_gap_min": int(runs.max()) if len(runs) else 0})
    return pd.DataFrame(rows)


def value_checks(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for node in values.columns:
        v = values[node]
        jump = v.diff().abs()
        rows.append({"node_id": node, "min": v.min(), "p50": v.median(), "p99": v.quantile(0.99), "max": v.max(),
                     "n_negative": int((v < 0).sum()), "n_zero": int((v == 0).sum()),
                     "zero_share": round(float((v == 0).mean()), 4),
                     f"n_jump_ge_{JUMP_M:g}m": int((jump >= JUMP_M).sum()), "max_jump": jump.max()})
    return pd.DataFrame(rows)


def build_report(nodes, info, d, agg, dup, intervals, values, mask, nm, vchk, elapsed) -> str:
    L = []
    w = L.append
    n_cells = mask.size
    w("[강남] 그래프 노드 수위 시계열 정렬 및 품질 리포트 (GNN 학습 없음)")
    w("=" * 66)
    w("※ 그래프 간선은 도로망 근접 기반 연구용 근사 관계이며 실제 하수관로 연결이 아니다.")
    w(f"입력: " + ", ".join(f"data/raw/{p.name}" for p in PERIOD_CSVS) + f" / 노드 순서: {NODES_CSV.name}")
    w(f"시간축: {START} ~ {END - pd.Timedelta(FREQ)} ({FREQ}, {len(mask):,}행) × 노드 {mask.shape[1]}열 "
      f"= {n_cells:,}칸")
    w("시간대: 원본 측정일자에 시간대 표기 없음 → 변환하지 않고 원본 시각(현지 시각으로 추정, KST 미확인) 그대로 사용")
    w(f"처리 시간 {elapsed:,.0f}초 / 추출 행 메모리 약 {d.memory_usage(deep=True).sum() / 1024**2:,.0f} MB, "
      f"행렬 {values.memory_usage().sum() / 1024**2:,.1f} MB (프로세스 최대 메모리는 측정 도구 없음)")

    w("\n1. ID 대응")
    found = set(d["node_id"].astype(str))
    w(f"  그래프 노드 {len(nodes)}개 중 원본에서 정확 일치로 발견: {len(found & set(nodes))} "
      f"/ 미발견: {sorted(set(nodes) - found) or '없음'}")
    w(f"  공백 등만 다른 근접 ID(사용 안 함): {info['near_miss_ids'] or '없음'}")
    w(f"  행렬 열 순서 = 그래프 idx 순서: {list(values.columns) == list(nodes)} → {list(values.columns)}")

    w("\n2. 레코드 수 대조")
    tot_t = sum(info["target_rows"].values())
    for m in info["file_rows"]:
        w(f"  {m}: 원본 {info['file_rows'][m]:,}행 중 대상 18개 노드 {info['target_rows'][m]:,}행")
    n_bad_t, n_out, n_off = int(d.bad_time.sum()), int((~d.in_period & ~d.bad_time).sum()), \
        int((d.in_period & ~d.on_grid).sum())
    n_cells_obs = int(mask.values.sum())
    n_nan_keys = int((agg["level"].isna() & ~agg["conflict"]).sum())
    w(f"  대상 행 합계 {tot_t:,} = 시각 파싱 실패 {n_bad_t:,} + 기간 밖 {n_out:,} + 1분 그리드 밖 {n_off:,} "
      f"+ 중복 초과행 {dup['dup_extra_rows']:,} + 값 충돌 칸 {dup['conflict_keys']:,} "
      f"+ 수위 빈값/비숫자 칸 {n_nan_keys:,} + 관측 칸 {n_cells_obs:,}")
    rhs = n_bad_t + n_out + n_off + dup["dup_extra_rows"] + dup["conflict_keys"] + n_nan_keys + n_cells_obs
    w(f"  → 설명된 행 {rhs:,}, 설명 안 된 차이 {tot_t - rhs:,}")
    w(f"  행렬 칸 {n_cells:,} = 관측 {n_cells_obs:,} + 결측 {n_cells - n_cells_obs:,} "
      "(결측 = 원본에 해당 시각 레코드 없음 + 값 충돌 + 수위 빈값/비숫자)")

    w("\n3. 시각 형식·측정 간격 (노드별 실측, 1분 가정 안 함)")
    w(f"  형식 '{TIME_FMT}' 파싱 실패 {n_bad_t:,}행, 초≠0 등 그리드 밖 {n_off:,}행")
    w(f"  원본 시각 범위: {d.time.min()} ~ {d.time.max()}")
    w("  노드별 간격 최빈값: " + ", ".join(f"{k:g}s: {v}" for k, v in intervals.interval_mode_s.value_counts().items()))
    w(f"  60초가 아닌 간격(결측 구간 포함): 노드당 최소 {intervals.n_not_60s.min():,}, 최대 {intervals.n_not_60s.max():,}")
    for r in intervals.itertuples():
        if r.n_not_60s:
            w(f"    {r.node_id}: 60s 비율 {r.share_60s:.4%}, 그 외 {r.other_intervals_top}")

    w("\n4. 중복·이상값 (값은 수정/삭제하지 않음)")
    w(f"  (노드, 시각) 중복 키 {dup['dup_keys']:,}개 (초과행 {dup['dup_extra_rows']:,}), 값 충돌 {dup['conflict_keys']:,}개 "
      "(충돌 칸은 결측 처리)")
    w(f"  수위 비숫자 {int(d.bad_level.sum()):,}행, 결측(빈 값) {int(d[LEVEL_COL].isna().sum()):,}행")
    w(f"  통신상태: {d[STATUS_COL].value_counts(dropna=False).to_dict()}")
    w(f"  기준(사전 고정): 음수, 0값 비율, 연속 1분 |Δ| ≥ {JUMP_M:g} m")
    w(vchk.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    w("\n5. 결측 현황")
    miss = 1 - mask.mean()
    w(f"  전체 결측률 {1 - mask.values.mean():.4%} ({n_cells - n_cells_obs:,}/{n_cells:,}칸)")
    w("  노드별 결측률 / 연속 결측 최대 길이(분) / 결측 구간 수:")
    for node in mask.columns:
        runs = gap_runs(mask[node].to_numpy())
        w(f"    {node}: {miss[node]:.4%} / {int(runs.max()) if len(runs) else 0:,} / {len(runs)}")
    w("  월별 결측률 (노드 평균, 최대 노드):")
    for m, g in nm.groupby("month"):
        w(f"    {m}: 평균 {g.missing_rate.mean():.4%}, 최대 {g.missing_rate.max():.4%} ({g.loc[g.missing_rate.idxmax(), 'node_id']})")
    all_runs = np.concatenate([gap_runs(mask[c].to_numpy()) for c in mask.columns])
    hist = pd.cut(pd.Series(all_runs), GAP_BINS, right=False, labels=GAP_LABELS).value_counts().reindex(GAP_LABELS)
    w("  연속 결측 구간 길이 분포(전 노드): " + ", ".join(f"{k} {v}" for k, v in hist.items()))
    w(f"  결측 칸 중 6분 이상 구간에 속한 비율: "
      f"{all_runs[all_runs >= 6].sum() / max(all_runs.sum(), 1):.1%}")
    w("  6분 이상 결측 구간 (시작 시각, 길이) — 고수위 시기와 겹치는지 확인용:")
    for node in mask.columns:
        x = np.diff(np.concatenate([[1], mask[node].to_numpy(), [1]]))
        spans = [(mask.index[s], e - s) for s, e in zip(np.where(x == -1)[0], np.where(x == 1)[0]) if e - s >= 6]
        if spans:
            w(f"    {node}: " + ", ".join(f"{t:%m-%d %H:%M} {n:,}분" for t, n in spans))
    daily = values.median(axis=1).resample("D").max()
    w(f"  18개 노드 중앙값 기준 일 최고수위 상위 3일: "
      + ", ".join(f"{t:%m-%d} {x:.3f} m" for t, x in daily.nlargest(3).items()))
    w(f"  노드별 최고수위 날짜: {values.idxmax().dt.strftime('%m-%d').value_counts().to_dict()}")
    all_obs = mask.all(axis=1)
    w(f"  18개 노드 모두 관측된 시각: {int(all_obs.sum()):,}/{len(mask):,} ({all_obs.mean():.2%})")
    w(f"  마스크 검증: 결측률 = 1 - mask 평균 = NaN 비율 → "
      f"{bool(((values.isna().values) == (mask.values == 0)).all())}")

    w("\n6. 결측 처리 정책 제안 (이번 단계 미적용)")
    w("  - 원본 행렬(NaN 유지)과 mask는 그대로 보존하고, 처리값은 별도 행렬 + 처리방법 코드 행렬로 저장")
    w("    (0=원본, 1=짧은 결측 전방 채움, 2=장기 결측 미처리). 처리 전후 값을 같은 시각 키로 추적")
    w("  - 짧은 결측(≤5분): 직전 관측값 전방 채움(LOCF) — 과거 정보만 사용, 선형보간처럼 미래값을 쓰지 않음")
    w("  - 긴 결측(>5분): 보간하지 않음. 학습 표본 생성 시 입력창/목표창에 포함되면 그 표본 제외 또는")
    w("    mask를 입력 채널로 넣어 모델이 결측을 인지하게 함. 0으로 채우지 않음(0은 실제 수위값으로 다수 존재)")
    w("  - 정규화 통계(평균·표준편차)는 학습 구간 관측값만으로 계산")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.perf_counter()
    nodes_tbl = pd.read_csv(NODES_CSV).sort_values("idx")
    nodes = nodes_tbl["node_id"].tolist()
    raw, info = extract_rows(PERIOD_CSVS, nodes)
    d = parse_rows(raw)
    agg, dup = duplicate_stats(d[d["in_period"] & d["on_grid"]])
    intervals = interval_stats(d)
    values, mask = build_matrix(agg, nodes)
    assert not values.index.duplicated().any() and list(values.columns) == nodes
    assert ((values.isna().values) == (mask.values == 0)).all()
    nm = node_month_quality(mask)
    vchk = value_checks(values)

    values.to_csv(OUT_VALUES, encoding="utf-8-sig", float_format="%.3f")
    mask.to_csv(OUT_MASK, encoding="utf-8-sig")
    np.savez_compressed(OUT_NPZ, values=values.to_numpy(np.float32), mask=mask.to_numpy(np.int8),
                        times=values.index.to_numpy().astype("datetime64[s]"),
                        node_ids=np.array(nodes, dtype=str), node_idx=nodes_tbl["idx"].to_numpy())
    with np.load(OUT_NPZ) as z:  # allow_pickle 없이 로드되는지 확인
        assert z["values"].shape == mask.shape and list(z["node_ids"]) == nodes
    nm.to_csv(OUT_NODE_MONTH, index=False, encoding="utf-8-sig")
    report = build_report(nodes, info, d, agg, dup, intervals, values, mask, nm, vchk, time.perf_counter() - t0)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
