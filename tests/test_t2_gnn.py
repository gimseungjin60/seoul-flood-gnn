"""실행: python -m tests.test_t2_gnn  (pytest로도 실행 가능). 실제 데이터 파일 불필요. 에폭을 줄여 합성 데이터로 검사한다.

설계안의 누수 테스트 4개 항목 + GNN 구조 점검:
 1) 타깃 시계열을 통째로/학습 구간만/평가 구간만 쓰레기·NaN 으로 바꿔도 예측이 비트 단위로 같다
 2) 같은 시드로 학습한 가중치가 타깃 시계열을 바꿔도 같다
 3) 특징 표준화 통계가 타깃 없이 계산된다(타깃 열·타깃의 강우 특징을 바꿔도 통계 불변)
 4) 평가 입력의 타깃 노드는 (상승분 0, 관측 플래그 0)이고, 어떤 노드도 타깃에서 메시지를 받지 않는다
"""
import numpy as np
import torch

from src.analysis.t2_gnn import (K_NEIGHBORS, build_adjacency, feature_stats, gnn_loso_predict, knn_weights, make_eval_inputs, prepare_target,
                                 train_model)

TR, EV = np.arange(0, 300), np.arange(300, 400)
I = 2
EPOCHS = 2


def _synthetic(n=8, b=400, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=b)) * 0.1
    R = common[:, None] + rng.normal(scale=0.3, size=(b, n))
    R[rng.random((b, n)) < 0.02] = np.nan
    rain_feat = [rng.normal(size=(b, 10)) for _ in range(n)]
    time_feat = rng.normal(size=(b, 3))
    pts = rng.uniform(0, 5000, size=(n, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(n)
    weights = np.where(rng.random(b) < 0.2, 5.0, 1.0)
    return R, dist, rain_feat, time_feat, weights


def _predict(R, dist, rf, tf, w, seeds=(0,)):
    return gnn_loso_predict(R, dist, rf, tf, I, TR, EV, w, seeds=seeds, epochs=EPOCHS)


def test_1_target_series_never_influences_prediction():
    R, dist, rf, tf, w = _synthetic()
    base = _predict(R, dist, rf, tf, w)
    variants = []
    a = R.copy()
    a[:, I] = np.random.default_rng(9).normal(size=len(R)) * 1000
    a[::3, I] = np.nan
    variants.append(a)                                                            # 전체 구간
    b = R.copy()
    b[TR, I] = -1e6
    variants.append(b)                                                            # 학습 구간만
    c = R.copy()
    c[EV, I] = 1e6
    variants.append(c)                                                            # 평가 구간만
    d = R.copy()
    d[:, I] = np.nan
    variants.append(d)                                                            # 전부 결측
    for v in variants:
        alt = _predict(v, dist, rf, tf, w)
        assert np.array_equal(base, alt, equal_nan=True), "타깃 시계열이 GNN 예측에 영향을 줌(누수)"


def test_2_trained_weights_identical_when_target_series_changes():
    R, dist, rf, tf, w = _synthetic(seed=1)
    R2 = R.copy()
    R2[:, I] = np.random.default_rng(5).normal(size=len(R)) * 100
    states = []
    for RR in (R, R2):
        c = prepare_target(RR, rf, tf, I)
        mu, sd = feature_stats(c["static_oth"], TR)
        A17 = build_adjacency(dist, c["others"], None)
        model = train_model(c["R_oth"], c["static_oth"], TR, A17, w, 0, mu, sd, EPOCHS)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), f"학습된 가중치 {k} 가 타깃 시계열에 의존"


def test_3_feature_stats_computed_without_target():
    R, dist, rf, tf, w = _synthetic(seed=2)
    c = prepare_target(R, rf, tf, I)
    mu, sd = feature_stats(c["static_oth"], TR)
    R2 = R.copy()
    R2[:, I] = 1e6
    rf2 = [x.copy() for x in rf]
    rf2[I] = rf2[I] + 100.0                                                       # 타깃의 강우 특징을 바꿔도 통계에는 안 들어간다
    c2 = prepare_target(R2, rf2, tf, I)
    mu2, sd2 = feature_stats(c2["static_oth"], TR)
    assert np.array_equal(mu, mu2) and np.array_equal(sd, sd2)
    assert c["static_oth"].shape[1] == len(rf) - 1 and c["R_oth"].shape[1] == R.shape[1] - 1     # 타깃 열이 없다


def test_4_eval_input_target_node_has_no_value_and_no_incoming_message():
    R, dist, rf, tf, w = _synthetic(seed=3)
    R2 = R.copy()
    R2[:, I] = 7.0
    c = prepare_target(R2, rf, tf, I)
    mu, sd = feature_stats(c["static_oth"], TR)
    x, m = make_eval_inputs(c["R_oth"], c["static_oth"], c["static_tgt"], EV, mu, sd)
    assert (x[:, -1, 0] == 0).all() and (m[:, -1] == 0).all()                    # 타깃 노드: 상승분 0, 관측 플래그 0
    A18 = build_adjacency(dist, c["others"], I)
    assert (A18[:, -1] == 0).all()                                               # 어떤 노드도 타깃에서 메시지를 받지 않는다
    A17 = build_adjacency(dist, c["others"], None)
    assert A17.shape == (len(rf) - 1, len(rf) - 1)


def test_graph_edges_k_nearest_inverse_distance_no_self():
    _, dist, *_ = _synthetic(n=10)
    cols = [j for j in range(10) if j != 3]
    W = knn_weights(dist, [3], cols)[0]
    assert (W > 0).sum() == K_NEIGHBORS
    nearest = sorted(cols, key=lambda c: dist[3, c])[:K_NEIGHBORS]
    for b, c in enumerate(cols):
        assert (W[b] > 0) == (c in nearest)
        if c in nearest:
            assert abs(W[b] - 1 / dist[3, c]) < 1e-12
    A = build_adjacency(dist, cols, None)
    assert (np.diag(A) == 0).all()


def test_not_vacuous_neighbors_change_prediction_and_seed_is_deterministic():
    R, dist, rf, tf, w = _synthetic(seed=4)
    base = _predict(R, dist, rf, tf, w)
    again = _predict(R, dist, rf, tf, w)
    assert np.array_equal(base, again, equal_nan=True)                            # 같은 시드 → 같은 결과
    R2 = R.copy()
    near = np.argsort(dist[I])[1]
    R2[EV, near] += 5.0                                                          # 가장 가까운 이웃의 평가 구간 값을 바꾸면 예측이 달라져야 한다
    assert not np.array_equal(base, _predict(R2, dist, rf, tf, w), equal_nan=True)


def test_all_neighbors_missing_gives_nan():
    R, dist, rf, tf, w = _synthetic(seed=5)
    R[EV[0], [j for j in range(R.shape[1]) if j != I]] = np.nan
    out = _predict(R, dist, rf, tf, w)
    assert np.isnan(out[0]) and np.isfinite(out[1:]).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_t2_gnn: OK")
