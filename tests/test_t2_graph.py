"""실행: python -m tests.test_t2_graph  (pytest로도 실행 가능)
그래프 보완 실험(G1 고도 방향, G2 랜덤)의 그래프 정의·누수·동치 테스트."""
import numpy as np
import torch

from src.analysis.t2_gnn import build_adjacency, gnn_loso_predict
from src.analysis.t2_graph import elev_adjacency, make_adjacency, rand_adjacency
from src.analysis.t2_graph_rules import K, elev_neighbors, nearest, q3_verdict, rand_neighbors, random_scores

rng = np.random.default_rng(0)
B, N, I = 900, 9, 2
TR, EV = np.arange(0, 700), np.arange(700, 900)
common = np.cumsum(rng.normal(size=B)) * 0.1
R0 = common[:, None] + rng.normal(scale=0.3, size=(B, N))
R0[rng.random((B, N)) < 0.2] = np.nan
pts = rng.uniform(0, 3000, size=(N, 2))
DIST = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(N)
ELEV = rng.uniform(10, 60, size=N)
RF = [rng.normal(size=(B, 10)) for _ in range(N)]
TF = rng.normal(size=(B, 3))
LABELS = [0, 1, 2, 3, 4]
W = np.ones(B)
OTHERS = [j for j in range(N) if j != I]


def test_road_adjacency_through_option_equals_default():
    road = make_adjacency(DIST, lambda _s: (lambda r, cols: nearest(DIST, r, cols)))
    for t in (None, I):
        assert np.allclose(road(OTHERS, t, 0), build_adjacency(DIST, OTHERS, t))


def test_elev_keeps_only_higher_within_road_knn_or_falls_back():
    for r in range(N):
        cols = [c for c in range(N) if c != r]
        near = nearest(DIST, r, cols)
        up, fb = elev_neighbors(DIST, ELEV, r, cols)
        assert set(up) <= set(near)
        if fb:
            assert up == near and all(ELEV[cols[b]] <= ELEV[r] for b in near)
        else:
            assert all(ELEV[cols[b]] > ELEV[r] for b in up)


def test_rand_graph_deterministic_k_and_no_self():
    s0, s0b, s1 = random_scores(N, 0), random_scores(N, 0), random_scores(N, 1)
    assert np.array_equal(s0, s0b) and not np.array_equal(s0, s1)
    for r in range(N):
        cols = list(range(N))
        nb = rand_neighbors(s0, r, cols)
        assert len(nb) == K and r not in [cols[b] for b in nb]


def test_target_column_zero_in_new_graphs():
    for adj in (elev_adjacency(DIST, ELEV), rand_adjacency(DIST)):
        A = adj(OTHERS, I, 0)
        assert A.shape == (N, N) and np.all(A[:, -1] == 0)
        assert np.all(A[:-1, :-1] == adj(OTHERS, None, 0))


def test_gnn_default_unchanged_and_target_never_influences():
    torch.set_num_threads(1)
    kw = dict(seeds=(0,), epochs=2, labels=LABELS)
    base = gnn_loso_predict(R0, DIST, RF, TF, I, TR, EV, W, **kw)
    road = make_adjacency(DIST, lambda _s: (lambda r, cols: nearest(DIST, r, cols)))
    assert np.array_equal(base, gnn_loso_predict(R0, DIST, RF, TF, I, TR, EV, W, adjacency=road, **kw), equal_nan=True)
    R1 = R0.copy()
    R1[:, I] = rng.normal(size=B) * 100
    for adj in (elev_adjacency(DIST, ELEV), rand_adjacency(DIST)):
        a = gnn_loso_predict(R0, DIST, RF, TF, I, TR, EV, W, adjacency=adj, **kw)
        assert np.array_equal(a, gnn_loso_predict(R1, DIST, RF, TF, I, TR, EV, W, adjacency=adj, **kw), equal_nan=True)


def test_q3_wording():
    assert q3_verdict(-1, 1) == "도로거리 구조의 기여 확인 안 됨"
    assert q3_verdict(-2, -1) == "도로거리 구조가 랜덤보다 나음"
    assert q3_verdict(1, 2) == "랜덤보다 나쁨"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
