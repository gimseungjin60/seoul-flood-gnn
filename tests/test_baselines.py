"""실행: python -m tests.test_baselines  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np
import pandas as pd

from src.data.forecast_inputs import make_inputs, make_targets, masked_mse, metrics, verify_samples
from src.models.baselines import eval_mask, last_observed, node_features, ridge_fit

T, N = 12, 2
VALUES = np.arange(T * N, dtype=float).reshape(T, N)
MASK = np.ones((T, N), dtype=bool)
MASK[3:7, 0] = False          # 노드0 결측 4분
MASK[9, 1] = False            # 노드1 정답 후보 결측
VALUES[~MASK] = np.nan


def test_inputs_have_no_nan_and_mark_imputed():
    xv, xo = make_inputs(VALUES, MASK, np.zeros(N), np.ones(N), np.array([6]), w=4)  # 창 [3,6]
    assert not np.isnan(xv).any()
    assert list(xo[0, :, 0]) == [0, 0, 0, 0] and list(xv[0, :, 0]) == [0, 0, 0, 0]  # 대체값은 obs=0으로 식별
    assert list(xo[0, :, 1]) == [1, 1, 1, 1] and xv[0, -1, 1] == VALUES[6, 1]


def test_masked_loss_ignores_missing_targets():
    y, ym = make_targets(VALUES, MASK, np.array([5, 7]), h=2)  # 정답 시각 7, 9
    assert ym.tolist() == [[True, True], [True, False]] and y[1, 1] == 0.0
    pred = np.full_like(y, 1e6)
    pred[ym] = y[ym] + 1.0
    assert masked_mse(pred, y, ym) == 1.0                    # 결측 정답의 1e6 오차는 무시
    assert masked_mse(pred, np.full_like(y, np.nan), np.zeros_like(ym)) == 0.0  # 유효 정답 0 → 0, NaN 없음
    assert metrics(pred, y, ym)["n_pairs"] == 3


def test_persistence_uses_window_only_and_no_fill():
    p = last_observed(VALUES, MASK, np.array([6, 8]), w=3)  # t=6 창 [4,6]: 노드0 관측 없음
    assert np.isnan(p[0, 0]) and p[0, 1] == VALUES[6, 1]
    assert p[1, 0] == VALUES[8, 0]
    p2 = last_observed(VALUES, MASK, np.array([6]), w=4)    # 창 [3,6]에도 없음(2는 창 밖) → NaN
    assert np.isnan(p2[0, 0])
    ym = np.array([[True, True]])
    assert eval_mask(ym, p[:1]).tolist() == [[False, True]]  # 예측 불가 쌍은 두 모델 공통으로 평가 제외


def test_node_features_use_only_own_node():
    zn = np.where(MASK, VALUES, 0.0)
    X = node_features(zn, MASK, np.array([11]), 1, w=3)
    assert X.shape == (1, 7)
    assert list(X[0, :3]) == [0.0, VALUES[10, 1], VALUES[11, 1]]  # 시각 9는 결측 → 0
    assert list(X[0, 3:6]) == [0.0, 1.0, 1.0] and X[0, 6] == 1.0
    zn2 = zn.copy()
    zn2[:, 0] = 999.0                                         # 다른 노드 바꿔도 특징 불변
    assert (node_features(zn2, MASK, np.array([11]), 1, w=3) == X).all()


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(200, 2)), np.ones((200, 1))])
    y = X @ np.array([2.0, -1.0, 0.5])
    assert np.allclose(ridge_fit(X, y, 1e-6), [2.0, -1.0, 0.5], atol=1e-4)


def test_verify_samples_catches_bad_counts():
    mask = np.ones((10, 1), dtype=bool)
    s = pd.DataFrame({"t_idx": [2, 3, 6], "split": ["train", "val", "test"], "n_target_observed": [1, 1, 1],
                      "strict_raw": [True, True, True], "strict_locf5": [True] * 3, "masked_any_target": [True] * 3})
    ok = verify_samples(s, mask, h=1, w=2)
    assert ok["strict_raw"] and ok["n_target_observed"]
    bad = s.assign(n_target_observed=[1, 0, 1])
    assert not verify_samples(bad, mask, h=1, w=2)["n_target_observed"]


if __name__ == "__main__":
    test_inputs_have_no_nan_and_mark_imputed()
    test_masked_loss_ignores_missing_targets()
    test_persistence_uses_window_only_and_no_fill()
    test_node_features_use_only_own_node()
    test_ridge_recovers_linear_relation()
    test_verify_samples_catches_bad_counts()
    print("ok")
