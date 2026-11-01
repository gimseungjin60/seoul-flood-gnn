"""실행: python -m tests.test_prepare_station_timeseries_2026  (pytest로도 실행 가능)"""
import numpy as np
import pandas as pd

import src.data.prepare_station_timeseries_2026 as ts


def rows(r) -> pd.DataFrame:
    t = pd.to_datetime(pd.Series([x[1] for x in r]), format=ts.SEWER_2026_TIME_FMT).dt.floor("min")
    return pd.DataFrame({"node_id": [x[0] for x in r], "time": t, "level": [x[2] for x in r]})


def test_time_format_parses_portal_style():
    t = pd.to_datetime(pd.Series(["01-JUN-26 00:00:37", "31-AUG-26 23:59:59"]), format=ts.SEWER_2026_TIME_FMT)
    assert t.tolist() == [pd.Timestamp("2026-06-01 00:00:37"), pd.Timestamp("2026-08-31 23:59:59")]


def test_minute_cells_floor_duplicates_conflicts():
    d = rows([
        ["A", "01-JUN-26 00:00:37", 0.10],
        ["A", "01-JUN-26 00:00:52", 0.10],     # 같은 분, 같은 값 → 1개
        ["A", "01-JUN-26 00:01:05", 0.10],
        ["A", "01-JUN-26 00:01:40", 0.50],     # 같은 분, 다른 값 → 충돌(결측)
        ["A", "01-JUN-26 00:02:10", 0.20],
        ["A", "01-JUN-26 00:02:50", np.nan],   # 값과 빈값 → 충돌 (2025판 nunique(dropna=False)와 같음)
        ["B", "01-JUN-26 00:00:00", 0.00],
    ])
    agg, st = ts.minute_cells(d)
    assert st == {"dup_keys": 3, "dup_extra_rows": 3, "conflict_keys": 2}
    a = agg.set_index(["node_id", "time"])["level"]
    assert a[("A", pd.Timestamp("2026-06-01 00:00"))] == 0.10
    assert pd.isna(a[("A", pd.Timestamp("2026-06-01 00:01"))])
    assert pd.isna(a[("A", pd.Timestamp("2026-06-01 00:02"))])
    assert a[("B", pd.Timestamp("2026-06-01 00:00"))] == 0.0   # 0은 실제 값


def test_matrix_and_selection_thresholds():
    n = 92 * 1440
    idx = pd.date_range(ts.START, ts.END, freq="1min", inclusive="left")
    full = np.ones(n, bool)
    cyc = (np.arange(n) % 120) < 45                     # 2시간 주기 45분 관측 → 블록 5/12 ≈ 0.42
    half = (np.arange(n) % 20) < 10                     # 10분 관측/10분 공백 → 블록 0.5, 분 0.5
    core_edge = np.arange(n) < int(np.ceil(0.90 * n))   # 분 커버리지 정확히 0.90
    parts = [pd.DataFrame({"node_id": k, "time": idx[m], "level": 0.1})
             for k, m in {"F": full, "C": cyc, "H": half, "E": core_edge}.items()]
    agg, _ = ts.minute_cells(pd.concat(parts, ignore_index=True))
    values, mask = ts.build_matrix(agg, ["C", "E", "F", "H", "Z"])
    assert len(values) == n and ((values.isna().values) == (mask.values == 0)).all()
    assert mask["Z"].sum() == 0
    loc = pd.DataFrame({"node_id": ["C", "E", "F", "H", "Z"], "location": "x", "n_location_variants": 1})
    sel = ts.select_sensors(values, mask, loc).set_index("node_id")
    assert sel.loc["F", "role"] == "core" and sel.loc["E", "role"] == "core"
    assert sel.loc["H", "role"] == "neighbor" and abs(sel.loc["H", "block_coverage"] - 0.5) < 1e-9
    assert sel.loc["C", "role"] == "excluded" and sel.loc["Z", "role"] == "excluded"
    assert sel.in_pool.tolist() == [False, True, True, True, False]


def test_quality_rules_sentinel_and_artifact_window():
    idx = pd.date_range("2026-07-07 13:00", periods=120, freq="1min")
    rng = np.random.default_rng(0)
    v = pd.DataFrame(0.1 + rng.integers(1, 9, (120, 12)) / 100, index=idx, columns=[f"S{i:02d}" for i in range(12)])
    v.iloc[30, 0] = -1.0                       # R1
    v.iloc[50:53, :8] = 2.0                    # R2: 12개 중 8개 정수 → 표시
    v.iloc[60, :7] = 0.0                       # 8분 뒤 다시 표시 → 같은 구간
    v.iloc[90, :5] = 1.0                       # 5/12 < 50% → 표시 안 함
    out, mask, info = ts.apply_quality_rules(v)
    assert info["sentinel_cells"] == {"S00": 1} and pd.isna(out.iloc[30, 0])
    assert info["flag_minutes"] == 4 and len(info["windows"]) == 1
    a, b, _ = info["windows"][0]
    assert (a, b) == (idx[45], idx[65])        # 50~60 + 앞뒤 5분
    assert out.loc[a:b].isna().all().all() and mask.loc[a:b].sum().sum() == 0
    assert out.iloc[90, 0] == 1.0 and out.iloc[44, 11] == v.iloc[44, 11]   # 구간 밖 값은 그대로
    assert ((out.isna().values) == (mask.values == 0)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
