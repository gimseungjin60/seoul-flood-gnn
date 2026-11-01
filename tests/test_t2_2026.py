"""실행: python -m tests.test_t2_2026  (pytest로도 실행 가능)
2026 일반화(labels: 가짜 타깃·GNN 손실을 core로 제한, H 실험 분리)의 누수·동치 테스트. 2025 동작은 tests/test_t2_loso.py, test_t2_gnn.py 가 그대로 검증한다."""
import numpy as np
import torch

from src.analysis.t2_2026 import h_setup
from src.analysis.t2_gnn import build_adjacency, feature_stats, gnn_loso_predict, prepare_target, train_model
from src.analysis.t2_loso import MODELS, Params, loso_predict

rng = np.random.default_rng(0)
B, N, I = 900, 7, 2
TR, EV = np.arange(0, 700), np.arange(700, 900)
common = np.cumsum(rng.normal(size=B)) * 0.1
R0 = common[:, None] + rng.normal(scale=0.3, size=(B, N))
R0[rng.random((B, N)) < 0.2] = np.nan
pts = rng.uniform(0, 3000, size=(N, 2))
DIST = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(N)
RF = [rng.normal(size=(B, 10)) for _ in range(N)]
TF = rng.normal(size=(B, 3))
LABELS = [0, 1, 2, 3]          # 4~6 은 이웃 입력 전용
W = np.ones(B)


def loso(R, labels):
    return loso_predict(R, DIST, RF, TF, I, TR, EV, Params(hgb_iter=5), labels=labels)


def test_loso_labels_target_series_never_influences():
    a = loso(R0, LABELS)
    R1 = R0.copy()
    R1[:, I] = rng.normal(size=B) * 100
    b = loso(R1, LABELS)
    for m in MODELS:
        assert np.array_equal(a[m], b[m], equal_nan=True), m


def test_loso_labels_none_equals_all_others():
    a = loso(R0, None)
    b = loso(R0, [j for j in range(N) if j != I])
    for m in MODELS:
        assert np.array_equal(a[m], b[m], equal_nan=True), m


def test_loso_labels_restrict_training_rows():
    """라벨이 아닌 노드의 '라벨로서의 값'만 바꿀 수는 없으므로, 라벨 집합이 바뀌면 학습 모델 예측이 달라지는지(라벨이 실제로 쓰이는지) 확인."""
    a, b = loso(R0, [0, 1]), loso(R0, [0, 1, 3, 4])
    assert not np.allclose(a["ridge_rain_time"], b["ridge_rain_time"], equal_nan=True)
    assert np.array_equal(a["idw_p1"], b["idw_p1"], equal_nan=True)     # IDW·평균은 학습이 없어 라벨과 무관


def gnn(R, labels, seeds=(0,)):
    return gnn_loso_predict(R, DIST, RF, TF, I, TR, EV, W, seeds=seeds, epochs=2, labels=labels)


def test_gnn_labels_target_series_never_influences():
    torch.set_num_threads(1)
    a = gnn(R0, LABELS)
    R1 = R0.copy()
    R1[:, I] = rng.normal(size=B) * 100
    assert np.array_equal(a, gnn(R1, LABELS), equal_nan=True)


def test_gnn_labels_none_equals_all_ones_mask():
    torch.set_num_threads(1)
    assert np.allclose(gnn(R0, None), gnn(R0, [j for j in range(N) if j != I]), equal_nan=True)


def test_gnn_zero_label_mask_never_updates_weights():
    torch.set_num_threads(1)
    c = prepare_target(R0, RF, TF, I)
    mu, sd = feature_stats(c["static_oth"], TR)
    A = build_adjacency(DIST, c["others"], None)
    torch.manual_seed(0)
    from src.analysis.t2_gnn import MaskAwareGNN
    init = MaskAwareGNN(2 + c["static_oth"].shape[-1]).state_dict()
    m = train_model(c["R_oth"], c["static_oth"], TR, A, W, 0, mu, sd, 2, label_mask=np.zeros(N - 1))
    assert all(torch.equal(init[k], v) for k, v in m.state_dict().items())


def test_h_setup_excludes_other_h_and_all_h_from_labels():
    roles = {"pool": ["a", "b", "h1", "c", "h2", "h3", "d"], "targets": ["a", "h1", "c", "h2", "h3"], "H": ["h1", "h2", "h3"]}
    nodes, i, labels = h_setup(roles, "h2")
    assert nodes == ["a", "b", "c", "h2", "d"] and nodes[i] == "h2"
    assert [nodes[j] for j in labels] == ["a", "c"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
