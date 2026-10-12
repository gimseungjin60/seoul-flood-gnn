"""실행: python -m tests.test_prepare_station_timeseries  (pytest로도 실행 가능)"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.prepare_station_timeseries as ts


def raw(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["node_id", "측정일자", "측정수위", "통신상태"]).assign(src_month="202506")


def test_extract_exact_ids_and_reports_near_miss():
    csv = ("unq_no,se_cd,se_nm,msrmt_ymd,msrmt_watl,sgn_stts,pstn_info\n"
           '"23-0001","23","강남","2025-06-01 00:00:00","0.10","통신양호","a"\n'
           '"23-0001 ","23","강남","2025-06-01 00:01:00","0.20","통신양호","a"\n'   # 공백 붙은 근접 ID
           '"23-0099","23","강남","2025-06-01 00:00:00","0.30","통신양호","b"\n')  # 대상 아님
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tv_swm_wal_mea_202506.csv"
        p.write_text(csv, encoding="utf-8")
        df, info = ts.extract_rows([p], ["23-0001"])
    assert len(df) == 1 and info["file_rows"]["202506"] == 3
    assert info["near_miss_ids"] == {repr("23-0001 "): 1}  # 사용하지 않고 보고만


def test_parse_duplicates_and_grid():
    d = ts.parse_rows(raw([
        ["A", "2025-06-01 00:00:00", "0.10", "통신양호"],
        ["A", "2025-06-01 00:00:00", "0.10", "통신양호"],   # 같은 값 중복
        ["A", "2025-06-01 00:01:00", "0.10", "통신양호"],
        ["A", "2025-06-01 00:01:00", "0.50", "통신양호"],   # 값 충돌
        ["A", "2025-06-01 00:02:30", "0.10", "통신양호"],   # 그리드 밖
        ["A", "2025-05-31 23:59:00", "0.10", "통신양호"],   # 기간 밖
        ["A", "bad", "0.10", "통신양호"],                   # 시각 파싱 실패
        ["A", "2025-06-01 00:03:00", "x", "통신양호"],      # 수위 비숫자
    ]))
    assert d["bad_time"].sum() == 1 and d["bad_level"].sum() == 1
    assert (d["in_period"] & ~d["on_grid"]).sum() == 1 and (~d["in_period"] & ~d["bad_time"]).sum() == 1
    agg, st = ts.duplicate_stats(d[d["in_period"] & d["on_grid"]])
    assert st == {"dup_keys": 2, "dup_extra_rows": 2, "conflict_keys": 1}
    assert pd.isna(agg.loc[agg["conflict"], "level"]).all()  # 충돌 값은 고르지 않음


def test_matrix_order_mask_and_no_fill():
    d = ts.parse_rows(raw([
        ["B", "2025-06-01 00:00:00", "0.00", "통신양호"],
        ["A", "2025-06-01 00:00:00", "0.20", "통신양호"],
        ["A", "2025-06-01 00:02:00", "0.30", "통신양호"],
    ]))
    agg, _ = ts.duplicate_stats(d)
    values, mask = ts.build_matrix(agg, ["A", "B", "C"])
    assert list(values.columns) == ["A", "B", "C"]                                   # 노드 idx 순서
    assert len(values) == 92 * 24 * 60 and not values.index.duplicated().any()      # 6/1~8/31 1분
    assert values.index[0] == pd.Timestamp("2025-06-01") and values.index[-1] == pd.Timestamp("2025-08-31 23:59")
    assert values.loc["2025-06-01 00:00", "B"] == 0.0 and mask.loc["2025-06-01 00:00", "B"] == 1  # 0은 실제 값
    assert pd.isna(values.loc["2025-06-01 00:01", "A"]) and mask.loc["2025-06-01 00:01", "A"] == 0  # 0으로 안 채움
    assert mask["C"].sum() == 0
    assert ((values.isna().values) == (mask.values == 0)).all()


def test_gap_runs_and_month_quality():
    assert list(ts.gap_runs(np.array([1, 0, 0, 1, 0, 1, 1, 0]))) == [2, 1, 1]
    assert list(ts.gap_runs(np.array([1, 1]))) == []
    idx = pd.date_range("2025-06-30 23:58", periods=4, freq="1min")
    mask = pd.DataFrame({"A": [1, 0, 0, 1]}, index=idx)
    q = ts.node_month_quality(mask).set_index("month")
    assert q.loc["2025-06", "missing"] == 1 and q.loc["2025-07", "max_gap_min"] == 1


if __name__ == "__main__":
    test_extract_exact_ids_and_reports_near_miss()
    test_parse_duplicates_and_grid()
    test_matrix_order_mask_and_no_fill()
    test_gap_runs_and_month_quality()
    print("ok")
