"""실행: python -m tests.test_build_forecast_dataset  (pytest로도 실행 가능)"""
import numpy as np
import pandas as pd

from src.data.build_forecast_dataset import (FILL_LOCF, FILL_MISSING, FILL_OBSERVED, build_samples, locf_causal,
                                             norm_stats, split_bounds, window_counts)

T = 20
TIMES = pd.date_range("2025-06-01", periods=T, freq="1min")


def test_input_target_indices_small_example():
    mask = np.ones((T, 2), dtype=bool)
    s = build_samples(mask, mask, TIMES, h=2, bounds={"train": (0, T)}, w=3)
    first, last = s.iloc[0], s.iloc[-1]
    # W=3, h=2 → 첫 anchor t=2: 입력 [0,2], 정답 4 / 마지막 anchor t=17: 입력 [15,17], 정답 19
    assert (first.t_idx, first.input_start, first.input_end, first.target_time) == (2, TIMES[0], TIMES[2], TIMES[4])
    assert (last.t_idx, last.input_start, last.target_time) == (17, TIMES[15], TIMES[19])
    assert len(s) == T - 3 - 2 + 1
    assert (s.target_time - s.input_end == pd.Timedelta("2min")).all()


def test_split_purges_boundary_samples():
    bounds = split_bounds(T)  # 70/15/15 → [0,14) [14,17) [17,20)
    assert bounds == {"train": (0, 14), "val": (14, 17), "test": (17, 20)}
    mask = np.ones((T, 1), dtype=bool)
    s = build_samples(mask, mask, TIMES, h=1, bounds=bounds, w=2)
    for name, (a, b) in bounds.items():
        x = s[s.split == name]
        assert ((x.t_idx - 1 >= a) & (x.t_idx + 1 < b)).all()  # 입력 시작·정답 모두 분할 안
    # t=13: 입력 [12,13] train, 정답 14 val → purged
    assert s.set_index("t_idx").loc[13, "split"] == "purged"


def test_locf_is_causal_and_tracked():
    v = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan], [np.nan], [np.nan], [np.nan]])
    filled, code = locf_causal(v, limit=2)
    assert list(filled[:, 0][:4]) == [1.0, 1.0, 1.0, 4.0]                  # 과거값(1.0)만 사용, 4.0 미사용
    assert list(code[:, 0]) == [FILL_OBSERVED, FILL_LOCF, FILL_LOCF, FILL_OBSERVED,
                                FILL_LOCF, FILL_LOCF, FILL_MISSING, FILL_MISSING]
    assert np.isnan(v[1, 0])                                                # 원본 미수정


def test_missing_schemes_and_target_uses_raw_mask():
    mask = np.ones((T, 2), dtype=bool)
    mask[5, 0] = False          # 입력 창 결측(1칸)
    mask[12, 1] = False         # 정답 결측
    filled_ok = mask.copy()
    filled_ok[5, 0] = True      # LOCF로 채워짐
    s = build_samples(mask, filled_ok, TIMES, h=1, bounds={"train": (0, T)}, w=3).set_index("t_idx")
    assert not s.loc[5, "strict_raw"] and s.loc[5, "strict_locf5"] and s.loc[5, "masked_any_target"]
    assert not s.loc[11, "strict_locf5"] and s.loc[11, "n_target_observed"] == 1  # 정답은 원본 mask 기준
    assert list(window_counts(mask, 3)[5]) == [2, 3]


def test_norm_uses_train_only():
    v = np.array([[1.0], [3.0], [100.0], [np.nan]])
    n = norm_stats(v, ["A"], train_end=2)
    assert n.loc[0, "train_mean"] == 2.0 and n.loc[0, "train_n_observed"] == 2


if __name__ == "__main__":
    test_input_target_indices_small_example()
    test_split_purges_boundary_samples()
    test_locf_is_causal_and_tracked()
    test_missing_schemes_and_target_uses_raw_mask()
    test_norm_uses_train_only()
    print("ok")
