"""실행: python -m tests.test_make_figures  (pytest로도 실행 가능). 그림 파일은 만들지 않는다."""
import numpy as np
import pandas as pd

from src.visualization.make_figures import bin_any_missing, label_offsets, select_representatives

IDX = pd.date_range("2025-06-01", periods=6, freq="1min")


def test_select_representatives_rules():
    values = pd.DataFrame({"A": [0.1, 0.1, 2.0, 0.1, 0.1, 0.1],     # 최고 수위
                           "B": [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],     # 0 m 최대
                           "C": [0.5, 0.5, 0.5, 0.5, 0.5, 0.51],    # 최소 변동
                           "D": [0.1, np.nan, np.nan, np.nan, 0.3, 0.2]}, index=IDX)  # 최장 결측
    mask = values.notna().astype(int)
    p = select_representatives(values, mask)
    assert p["A"] == "최고 수위" and p["B"] == "0 m 비율 최대" and p["C"] == "최소 변동"
    # D는 '표준편차 중앙값'과 '최장 결측' 두 규칙에 모두 해당 → 중복 없이 먼저 맞은 규칙 라벨 하나만
    assert p["D"] == "전형적 변동(표준편차 중앙값)" and len(p) == 4


def test_bin_any_missing_keeps_single_minute_gap():
    mask = pd.DataFrame({"A": [1, 1, 0, 1, 1, 1], "B": [1] * 6}, index=IDX)
    b = bin_any_missing(mask, "3min")
    assert b["A"].tolist() == [0, 1] and b["B"].tolist() == [1, 1]


def test_label_offsets_follow_geography():
    nodes = pd.DataFrame({"node_id": ["W", "E", "N", "S", "far"],
                          "lon": [127.000, 127.002, 127.050, 127.050, 127.100],
                          "lat": [37.500, 37.500, 37.512, 37.510, 37.400]})
    off = label_offsets(nodes)
    assert off["W"][0] < 0 < off["E"][0]       # 서쪽 라벨은 왼쪽, 동쪽은 오른쪽
    assert off["N"][1] > 0 > off["S"][1]       # 북쪽 위, 남쪽 아래
    assert off["far"] == (6, 6)


if __name__ == "__main__":
    test_select_representatives_rules()
    test_bin_any_missing_keeps_single_minute_gap()
    test_label_offsets_follow_geography()
    print("ok")
