"""실행: python -m tests.test_lead_lag  (pytest로도 실행 가능). 센서 간 시차 분석 규칙 함수 테스트."""
import numpy as np

from src.analysis.lead_lag_rules import (consistent_fraction, first_cross, is_consistent, pair_summary, rain_t50_block,
                                         sign_flip_null, smooth_rise)


def test_smooth_rise_mm_and_missing():
    lv = np.full(30, 1.0)
    lv[10:] = 1.2
    sm = smooth_rise(lv, 1.0)
    assert np.isclose(sm[25], 200.0)
    assert np.isclose(sm[2], 0.0)
    lv2 = np.full(30, np.nan)
    lv2[0] = 1.0
    assert np.isnan(smooth_rise(lv2, 1.0)[15])


def test_first_cross_half_of_peak():
    x = np.array([0, 10, 40, 60, 100, 80, np.nan])
    assert first_cross(x) == 3
    assert first_cross(np.full(3, np.nan)) is None


def test_rain_t50_block():
    assert rain_t50_block(np.array([0, 1, 1, 2, 0])) == 2
    assert rain_t50_block(np.zeros(4)) is None


def test_pair_summary_and_consistency():
    m, c = pair_summary(np.array([20.0, 15.0, 30.0, -5.0]))
    assert m == 17.5 and c == 0.75 and is_consistent(m, c)
    assert not is_consistent(*pair_summary(np.array([5.0, 6.0, 7.0, 8.0])))   # 10분 미만
    assert pair_summary(np.array([0.0, 0.0, 0.0, 0.0])) == (0.0, 0.0)


def test_sign_flip_null_removes_systematic_lead():
    pairs = [np.array([20.0, 25.0, 30.0, 22.0, 18.0, 27.0])] * 50
    assert consistent_fraction(pairs) == 1.0
    null = sign_flip_null(pairs, n_perm=200, seed=0)
    assert np.percentile(null, 95) < 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
