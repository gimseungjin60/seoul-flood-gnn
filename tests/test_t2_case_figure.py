"""실행: python -m tests.test_t2_case_figure  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np

from src.analysis.t2_case_figure import normalize_window
from src.analysis.t2_loso_sensitivity import subset


def test_normalize_window_maps_min_to_zero_and_max_to_one_and_handles_constant_and_nan():
    x = np.array([2.0, 4.0, np.nan, 6.0])
    z = normalize_window(x)
    assert z[0] == 0.0 and z[3] == 1.0 and z[1] == 0.5 and np.isnan(z[2])
    assert (normalize_window(np.array([3.0, 3.0, 3.0])) == 0.0).all()                # 상수 구간은 0으로 나누지 않는다


def test_subset_drops_the_requested_sensor_and_events_only():
    E, J, M = 4, 3, 2
    S = np.arange(E * J * M, dtype=float).reshape(E, J, M)
    N = np.ones((E, J))
    S2, N2, ev2, nd2 = subset(S, N, [1, 5, 8, 9], ["a", "b", "c"], drop_sensor="b", drop_events=(5, 8))
    assert S2.shape == (2, 2, M) and ev2 == [1, 9] and nd2 == ["a", "c"]
    assert np.array_equal(S2[0, :, 0], S[0, [0, 2], 0]) and np.array_equal(S2[1, :, 1], S[3, [0, 2], 1])


if __name__ == "__main__":
    test_normalize_window_maps_min_to_zero_and_max_to_one_and_handles_constant_and_nan()
    test_subset_drops_the_requested_sensor_and_events_only()
    print("ok")
