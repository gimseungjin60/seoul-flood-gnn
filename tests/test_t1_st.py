"""실행: python -m tests.test_t1_st  (pytest로도 실행 가능)
T1 시공간 예측의 누수 테스트: 기점 t 의 예측(모든 모델)은 t 이후 블록 값에 영향받지 않아야 하고, 상관 그래프는 학습 구간만 쓴다."""
import numpy as np
import pandas as pd
import torch

from src.analysis.spatial_signal_profile_adjusted import hour_of_week
from src.analysis.t1_st import corr_graph, history, run_fold, targets
from src.analysis.t1_st_rules import HORIZONS, WINDOW
from src.analysis.t2_loso import fold_slices, time_features

rng = np.random.default_rng(0)
B, N = 1500, 7
TG = [0, 1, 2, 3]
bt = pd.date_range("2026-06-01 00:09", periods=B, freq="10min")
common = np.cumsum(rng.normal(size=B)) * 0.01
BLK = 0.2 + common[:, None] + rng.normal(scale=0.01, size=(B, N))
BLK[rng.random((B, N)) < 0.1] = np.nan
pts = rng.uniform(0, 3000, size=(N, 2))
INP = dict(blk=BLK, bt=bt, how=hour_of_week(bt - pd.Timedelta(minutes=9)), dist=np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(N),
           rain_feat=[rng.normal(size=(B, 10)) for _ in range(N)], time_feat=time_features(bt),
           events=pd.DataFrame({"event_id": [1], "start": [bt[900]], "end": [bt[950]]}))
FOLD = list(fold_slices(B))[2]


def run(inp):
    eval_idx, _, train_mask = FOLD
    out, scale, _ = run_fold(inp, TG, eval_idx, train_mask, seeds=(0,), epochs=1)
    return out


def test_history_uses_only_past():
    a = rng.normal(size=(50, 3))
    val, obs = history(a)
    assert np.allclose(val[10, :, -1], a[10]) and np.allclose(val[10, :, 0], a[10 - WINDOW + 1])
    assert obs[: WINDOW - 1, :, 0].sum() == 0
    y, ok = targets(a)
    assert np.allclose(y[10, :, 1], a[10 + HORIZONS[1]] - a[10]) and not ok[-1].any()


def test_future_values_never_change_past_origin_predictions():
    torch.set_num_threads(1)
    eval_idx = FOLD[0]
    t0 = eval_idx[len(eval_idx) // 2]
    a = run(INP)
    blk2 = BLK.copy()
    blk2[t0 + 1: eval_idx[-1] + 1] = rng.normal(size=(eval_idx[-1] - t0, N)) * 5
    b = run({**INP, "blk": blk2})
    k = np.searchsorted(eval_idx, t0) + 1
    for m in a:
        assert np.allclose(a[m][:k], b[m][:k], equal_nan=True), m
        assert not np.allclose(a[m][k:], b[m][k:], equal_nan=True), m    # 이후 기점은 실제로 바뀐다(테스트가 의미 있음)


def test_corr_graph_ignores_eval_period():
    _, _, train_mask = FOLD
    a = BLK - np.nanmean(BLK, 0)
    a2 = a.copy()
    a2[~train_mask] = rng.normal(size=((~train_mask).sum(), N))
    assert np.array_equal(corr_graph(a, train_mask), corr_graph(a2, train_mask))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
