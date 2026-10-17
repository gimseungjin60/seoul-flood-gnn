"""실행: python -m tests.test_spatial_signal  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np

from src.analysis.spatial_signal import block_mean, loo_eval, mantel, wmean


def test_block_mean_requires_min_obs():
    v = np.array([[1.0], [2.0], [3.0], [np.nan], [np.nan], [5.0], [np.nan], [np.nan], [np.nan], [np.nan]])
    m = block_mean(v, b=5, min_obs=3)
    assert m.shape == (2, 1) and m[0, 0] == 2.0 and np.isnan(m[1, 0])   # 둘째 블록 관측 1개 → 무효


def test_wmean_ignores_nan_neighbors():
    a = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    out = wmean(a, np.array([1.0, 5.0, 1.0]))
    assert out[0] == 2.0 and np.isnan(out[1])


def test_mantel_detects_distance_decay():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, size=(12, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    corr = np.exp(-dist / 3)
    rho, p = mantel(dist, corr, n_perm=200)
    assert rho < -0.95 and p < 0.05
    rho2, p2 = mantel(dist, rng.uniform(size=dist.shape), n_perm=200)
    assert p2 > 0.05


def test_loo_uses_neighbors_and_never_target_itself():
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(size=400)) * 0.01
    x = np.stack([common + 1.0, common + 2.0, common + 3.0], axis=1)   # 세 노드가 같은 변동을 공유
    dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    loo = loo_eval(x, dist, 300, 400, ["a", "b", "c"], n_perm=3)
    mae = loo[(loo.subset == "all") & ~loo.model.str.startswith("perm")].groupby("model")["MAE"].mean()
    assert mae["idw_p1"] < 1e-9 and mae["mean_others"] < 1e-9 and mae["const"] > 1e-3
    x2 = x.copy()
    x2[300:, 0] += 5.0                                                  # 대상 노드 정답이 바뀌어도 추정은 그대로여야 한다
    loo2 = loo_eval(x2, dist, 300, 400, ["a", "b", "c"], n_perm=3)
    a = loo[(loo.node == "a") & (loo.model == "idw_p1") & (loo.subset == "all")]["MAE"].iloc[0]
    b = loo2[(loo2.node == "a") & (loo2.model == "idw_p1") & (loo2.subset == "all")]["MAE"].iloc[0]
    assert abs(b - a) > 4.0                                             # 오차만 커지고(정답 비교) 추정 입력은 이웃뿐


if __name__ == "__main__":
    test_block_mean_requires_min_obs()
    test_wmean_ignores_nan_neighbors()
    test_mantel_detects_distance_decay()
    test_loo_uses_neighbors_and_never_target_itself()
    print("ok")
