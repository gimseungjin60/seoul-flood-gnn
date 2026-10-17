"""실행: python -m tests.test_t2_loso  (pytest로도 실행 가능). 실제 데이터 파일 불필요.

핵심: 타깃 센서의 시계열이 어떤 모델의 예측에도 영향을 주지 않는다(누수 없음)."""
import numpy as np

from src.analysis.t2_loso import Params, fold_slices, loso_predict, neighbor_features, nowcast_rain_features, MODELS, N_FOLDS, PURGE


def _synthetic(n=5, b=900, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=b)) * 0.1
    R = common[:, None] + rng.normal(scale=0.3, size=(b, n)) if informative else rng.normal(size=(b, n))
    R[rng.random((b, n)) < 0.02] = np.nan                                       # 일부 결측
    rain_feat = [rng.normal(size=(b, 10)) for _ in range(n)]
    time_feat = rng.normal(size=(b, 3))
    pts = rng.uniform(0, 5000, size=(n, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(n)         # 대각 0 방지(1m)
    return R, dist, rain_feat, time_feat


def _run(R, dist, rain_feat, time_feat, i=2):
    return loso_predict(R, dist, rain_feat, time_feat, i, np.arange(0, 700), np.arange(700, 900), Params(hgb_iter=5))


def test_target_series_never_influences_any_prediction():
    R, dist, rf, tf = _synthetic()
    base = _run(R, dist, rf, tf)
    R2 = R.copy()
    R2[:, 2] = np.random.default_rng(9).normal(size=len(R)) * 1000               # 타깃 열을 통째로 쓰레기로
    R2[::3, 2] = np.nan
    alt = _run(R2, dist, rf, tf)
    for m in MODELS:
        assert np.array_equal(base[m], alt[m], equal_nan=True), f"타깃 시계열이 {m} 예측에 영향을 줌(누수)"


def test_target_series_in_training_and_eval_periods_both_ignored():
    R, dist, rf, tf = _synthetic(seed=3)
    base = _run(R, dist, rf, tf)
    R2 = R.copy()
    R2[700:, 2] = 1e6                                                            # 평가 구간만 변경
    R3 = R.copy()
    R3[:700, 2] = -1e6                                                           # 학습 구간만 변경
    for alt in (_run(R2, dist, rf, tf), _run(R3, dist, rf, tf)):
        for m in MODELS:
            assert np.array_equal(base[m], alt[m], equal_nan=True)


def test_test_is_not_vacuous_other_sensors_do_change_predictions():
    R, dist, rf, tf = _synthetic()
    base = _run(R, dist, rf, tf)
    R2 = R.copy()
    R2[:, 0] += 5.0                                                              # 이웃 센서를 바꾸면 이웃 사용 모델은 달라져야 한다
    alt = _run(R2, dist, rf, tf)
    assert not np.array_equal(base["mean_others"], alt["mean_others"], equal_nan=True)
    assert not np.array_equal(base["ridge_rain_time_nbr"], alt["ridge_rain_time_nbr"], equal_nan=True)


def test_rain_time_models_ignore_neighbors_entirely():
    R, dist, rf, tf = _synthetic()
    base = _run(R, dist, rf, tf)
    R2 = R.copy()
    R2[:, 0] += 5.0
    alt = _run(R2, dist, rf, tf)
    # (3)은 '타깃 지점 강우+시간대'만 입력이지만, 가짜 타깃의 라벨(이웃 센서 값)을 학습에 쓰므로 계수는 달라질 수 있다.
    # 평가 시점의 입력이 이웃 값에 의존하지 않음을 확인: 같은 학습 결과라면 이웃 값이 바뀐 평가 구간에서도 예측이 같아야 한다.
    R3 = R.copy()
    R3[700:, 0] += 5.0                                                           # 평가 구간의 이웃 값만 변경
    alt3 = _run(R3, dist, rf, tf)
    assert np.array_equal(base["ridge_rain_time"], alt3["ridge_rain_time"], equal_nan=True)
    assert np.array_equal(base["hgb_rain_time"], alt3["hgb_rain_time"], equal_nan=True)
    assert not np.array_equal(base["ridge_rain_time_nbr"], alt3["ridge_rain_time_nbr"], equal_nan=True)


def test_neighbors_help_when_signal_is_shared():
    R, dist, rf, tf = _synthetic(informative=True, seed=1)
    out = _run(R, dist, rf, tf)
    a = R[700:, 2]
    ok = np.isfinite(a) & np.isfinite(out["ridge_rain_time_nbr"])
    e3 = np.abs(out["ridge_rain_time"][ok] - a[ok]).mean()
    e4 = np.abs(out["ridge_rain_time_nbr"][ok] - a[ok]).mean()
    assert e4 < 0.7 * e3                                                         # 공통 신호가 있으면 이웃을 쓴 쪽이 크게 낫다


def test_neighbor_features_exclude_missing_and_handle_all_missing_rows():
    R_c = np.array([[1.0, np.nan, 3.0, 5.0], [np.nan, np.nan, np.nan, np.nan]])
    d = np.array([100.0, 200.0, 300.0, 400.0])
    f, nv = neighbor_features(R_c, d)
    assert nv.tolist() == [3, 0] and np.isfinite(f).all()                        # 유효 이웃 0이어도 NaN 없이 채움(행은 호출 쪽에서 제외)


def test_nowcast_rain_uses_same_slot_and_rejects_short_history():
    rain = np.zeros(400)
    rain[300] = 4.0
    f = nowcast_rain_features(rain, np.array([300, 301]))
    assert np.isclose(f[0, 0], np.log1p(4.0)) and np.isclose(f[1, 1], np.log1p(4.0))   # 같은 슬롯 포함, 한 칸 뒤 블록은 1슬롯 전
    try:
        nowcast_rain_features(rain, np.array([100]))
    except ValueError:
        return
    raise AssertionError("이전 슬롯 부족이 거부되지 않음")


def test_fold_purge_removes_buffer_around_eval_fold():
    B = 2000
    for eval_idx, train_idx, tr in fold_slices(B):
        a, b = eval_idx[0], eval_idx[-1] + 1
        assert not tr[a:b].any()
        assert not tr[max(0, a - PURGE): a].any() and not tr[b: min(B, b + PURGE)].any()
        assert len(train_idx) == tr.sum()
    assert len(list(fold_slices(B))) == N_FOLDS


if __name__ == "__main__":
    test_target_series_never_influences_any_prediction()
    test_target_series_in_training_and_eval_periods_both_ignored()
    test_test_is_not_vacuous_other_sensors_do_change_predictions()
    test_rain_time_models_ignore_neighbors_entirely()
    test_neighbors_help_when_signal_is_shared()
    test_neighbor_features_exclude_missing_and_handle_all_missing_rows()
    test_nowcast_rain_uses_same_slot_and_rejects_short_history()
    test_fold_purge_removes_buffer_around_eval_fold()
    print("ok")
