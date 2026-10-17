"""실행: python -m tests.test_spatial_signal_rain_adjusted  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np

from src.analysis.spatial_signal_rain_adjusted import rain_design, remove_rain


def test_rain_design_lags_and_nan_head():
    r = np.array([0.0, 1.0, 3.0, 7.0])
    X = rain_design(r)
    assert X.shape == (4, 3) and np.isnan(X[0, 1]) and np.isnan(X[1, 2])
    assert np.isclose(X[3, 0], np.log1p(7.0)) and np.isclose(X[3, 1], np.log1p(3.0)) and np.isclose(X[3, 2], np.log1p(1.0))


def test_remove_rain_removes_linear_effect_and_keeps_independent_signal():
    rng = np.random.default_rng(0)
    n = 3000
    rain = np.where(rng.random(n) < 0.2, rng.uniform(0, 10, n), 0.0)
    other = rng.normal(size=n) * 0.1                                  # 강우와 무관한 신호
    dy = 0.3 * np.log1p(rain) + 0.1 * np.roll(np.log1p(rain), 1) + other
    fit_rows = np.arange(n) < 2000
    res, r2 = remove_rain(dy, rain, fit_rows)
    ok = np.isfinite(res)
    assert r2 > 0.5 and abs(np.corrcoef(res[ok], other[ok])[0, 1]) > 0.95       # 강우 성분이 빠지고 무관 신호가 남음
    assert abs(np.corrcoef(res[ok], np.log1p(rain)[ok])[0, 1]) < 0.1


def test_remove_rain_fits_only_on_fit_rows():
    n = 200
    rain = np.zeros(n)
    rain[100:] = 5.0
    dy = np.where(np.arange(n) < 100, 0.0, 1.0)                        # 강우 이후에만 dy=1
    res_all, _ = remove_rain(dy, rain, np.ones(n, dtype=bool))
    res_pre, _ = remove_rain(dy, rain, np.arange(n) < 100)             # 강우가 없던 구간만으로 적합 → 효과를 배우지 못함
    assert np.nanmax(np.abs(res_all)) < 1e-6 and np.nanmax(np.abs(res_pre[105:])) > 0.5


if __name__ == "__main__":
    test_rain_design_lags_and_nan_head()
    test_remove_rain_removes_linear_effect_and_keeps_independent_signal()
    test_remove_rain_fits_only_on_fit_rows()
    print("ok")
