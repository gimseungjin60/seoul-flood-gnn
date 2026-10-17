"""실행: python -m tests.test_assign_rain_gauges  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np
import pandas as pd

from src.data.assign_rain_gauges import assign_nearest, best_lag_corr, haversine_matrix, node_blocks


def test_haversine_known_distance():
    d = haversine_matrix(np.array([37.5]), np.array([127.0]), np.array([37.5, 37.51]), np.array([127.0, 127.0]))
    assert d[0, 0] == 0.0 and abs(d[0, 1] - 1111.9) < 2          # 위도 0.01도 = 약 1,112 m


def test_assign_nearest_picks_closest_not_same_district():
    nodes = pd.DataFrame({"node_id": ["a"], "lat": [37.500], "lon": [127.000]})
    loc = pd.DataFrame({"gu": ["강남구", "서초구"], "gname": ["far", "near"], "gid": ["1", "2"],
                        "lat": [37.520, 37.501], "lon": [127.000, 127.000]})
    r = assign_nearest(nodes, loc)
    assert r.gauge_name[0] == "near" and r.gauge_gu[0] == "서초구" and r.second_gauge_name[0] == "far"


def test_node_blocks_nan_when_few_observations():
    v = np.ones((20, 1))
    m = np.ones((20, 1), dtype=int)
    m[:7, 0] = 0                                                   # 첫 블록 관측 3개 → 무효(최소 5개)
    b = node_blocks(v, m)
    assert np.isnan(b[0, 0]) and b[1, 0] == 1.0


def test_best_lag_corr_finds_shift():
    rng = np.random.default_rng(0)
    rain = np.where(rng.random(1000) < 0.15, rng.uniform(1, 5, 1000), 0.0)
    dy = np.roll(rain, 1)                                          # 수위가 10분(1칸) 뒤에 반응
    c, lag = best_lag_corr(rain, dy)
    assert lag == 10 and c > 0.99


if __name__ == "__main__":
    test_haversine_known_distance()
    test_assign_nearest_picks_closest_not_same_district()
    test_node_blocks_nan_when_few_observations()
    test_best_lag_corr_finds_shift()
    print("ok")
