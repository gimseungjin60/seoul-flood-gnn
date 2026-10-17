"""실행: python -m tests.test_t2_loso_bootstrap  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np
import pandas as pd

from src.analysis.t2_loso import MODELS
from src.analysis.t2_loso_bootstrap import bootstrap, ci, event_error_sums, macro_mae


def _toy(E=8, J=4, seed=0):
    rng = np.random.default_rng(seed)
    N = rng.integers(20, 60, size=(E, J)).astype(float)
    S = np.zeros((E, J, len(MODELS)))
    base = rng.uniform(0.01, 0.03, size=(E, J))
    for k in range(len(MODELS)):
        S[:, :, k] = base * N * (1.0 + 0.0 * k)
    return S, N


def test_macro_mae_with_all_events_once_equals_pooled_mae():
    S, N = _toy()
    point, per = macro_mae(S, N, np.ones((1, S.shape[0])))
    pooled = S.sum(axis=0) / N.sum(axis=0)[:, None]               # 센서별 풀링 MAE
    assert np.allclose(per[0], pooled) and np.allclose(point[0], pooled.mean(axis=0))


def test_bootstrap_difference_ci_excludes_zero_for_consistently_better_model_and_includes_zero_for_identical():
    S, N = _toy()
    k = {m: i for i, m in enumerate(MODELS)}
    S[:, :, k["idw_p1"]] *= 0.8                                    # IDW 가 모든 이벤트·센서에서 20% 낮은 오차
    macro, _ = bootstrap(S, N, n_boot=500, seed=1)
    d = macro[:, k["ridge_rain_time_nbr"]] - macro[:, k["idw_p1"]]
    lo, hi = ci(d)
    assert lo > 0                                                  # 차이 CI 가 0 을 포함하지 않음
    same = macro[:, k["mean_others"]] - macro[:, k["ridge_rain_time"]]
    lo2, hi2 = ci(same)
    assert lo2 <= 0 <= hi2                                         # 동일한 두 모델은 0 을 포함(구분 불가)


def test_overlapping_event_windows_are_rejected():
    bt = pd.date_range("2025-06-01 00:09", periods=400, freq="10min")
    ev = pd.DataFrame({"event_id": [1, 2], "start": [bt[10], bt[20]], "end": [bt[30], bt[60]]})   # 창이 겹침
    pred = {m: np.zeros((400, 2)) for m in MODELS}
    try:
        event_error_sums(pred, np.zeros((400, 2)), bt, ev)
    except ValueError:
        return
    raise AssertionError("겹치는 이벤트 창이 거부되지 않음")


def test_event_error_sums_counts_only_valid_blocks():
    bt = pd.date_range("2025-06-01 00:09", periods=400, freq="10min")
    ev = pd.DataFrame({"event_id": [1], "start": [bt[10]], "end": [bt[30]]})
    actual = np.zeros((400, 1))
    actual[15, 0] = np.nan                                         # 결측 블록은 제외
    pred = {m: np.ones((400, 1)) for m in MODELS}
    S, N, ids = event_error_sums(pred, actual, bt, ev)
    n_blocks = int(((bt >= bt[10]) & (bt <= bt[30] + pd.Timedelta(hours=3))).sum())
    assert ids == [1] and N[0, 0] == n_blocks - 1 and np.isclose(S[0, 0, 0], n_blocks - 1)


if __name__ == "__main__":
    test_macro_mae_with_all_events_once_equals_pooled_mae()
    test_bootstrap_difference_ci_excludes_zero_for_consistently_better_model_and_includes_zero_for_identical()
    test_overlapping_event_windows_are_rejected()
    test_event_error_sums_counts_only_valid_blocks()
    print("ok")
