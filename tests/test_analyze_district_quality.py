"""실행: python -m tests.test_analyze_district_quality  (pytest로도 실행 가능)"""
import numpy as np
import pandas as pd

from src.data.analyze_district_quality import node_coverage, spatial_summary


def test_node_coverage():
    start, end = pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-01 01:40")  # 100분
    t0 = start.value // 10**9
    a = t0 + 60 * np.arange(100)                      # 60s, 100%
    b = t0 + 300 * np.arange(10)                      # 300s, 10/20 = 50%
    c = np.array([t0 - 60, end.value // 10**9])       # 기간 밖 → 제외
    ids = np.concatenate([np.zeros(100), np.ones(11), [2, 2]]).astype(int)
    secs = np.concatenate([a, b, [b[0]], c])          # B 중복 시각 1건
    q = node_coverage(ids, secs, start, end)
    assert q.loc[0, "interval_mode_s"] == 60 and q.loc[0, "coverage_ratio"] == 1.0
    assert q.loc[1, "measurement_count"] == 10 and q.loc[1, "coverage_ratio"] == 0.5
    assert 2 not in q.index


def test_spatial_summary():
    s = spatial_summary(pd.Series(["강남구 압구정로20길 6앞 맨홀", "강남구 압구정로 128",
                                   "강남구 역삼동 820-5", "강남구 테헤란로 447", None]))
    assert s["n_roads"] == 2 and s["n_dongs"] == 1
    assert s["top_key"] == "압구정로" and s["top_share"] == 0.5


if __name__ == "__main__":
    test_node_coverage()
    test_spatial_summary()
    print("ok")
