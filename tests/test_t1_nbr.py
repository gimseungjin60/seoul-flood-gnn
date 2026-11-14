"""실행: python -m tests.test_t1_nbr  (pytest로도 실행 가능)
G-none 테스트: 0 인접 행렬이면 다른 센서 값이 예측에 전혀 안 섞임, 그래프 교체가 원래대로 돌아옴, 확장 설정에서 누수 없음."""
import numpy as np
import torch

from src.analysis import t1_st
from src.analysis.t1_long import long_settings
from src.analysis.t1_nbr import no_neighbors
from src.analysis.t1_nbr_rules import n1_verdict, r1_verdict
from tests import test_t1_st as base


def test_graph_swap_restored():
    old = t1_st.road_graph
    with no_neighbors():
        assert not t1_st.road_graph(np.ones((4, 4))).any()
    assert t1_st.road_graph is old


def test_zero_graph_means_no_neighbor_influence():
    torch.manual_seed(0)
    model = t1_st.STGNN(3, n_out=2).eval()
    seq, stat, m = torch.randn(2, 5, 12, 2), torch.randn(2, 5, 3), torch.ones(2, 5)
    seq2 = seq.clone()
    seq2[:, 1:] = torch.randn(2, 4, 12, 2) * 10                 # 센서 0 이외 값만 바꿈
    A = torch.zeros(5, 5)
    with torch.no_grad():
        assert torch.allclose(model(seq, stat, m, A)[:, 0], model(seq2, stat, m, A)[:, 0], atol=1e-6)
        assert not torch.allclose(model(seq, stat, m, torch.ones(5, 5))[:, 0], model(seq2, stat, m, torch.ones(5, 5))[:, 0])


def test_verdict_labels():
    assert n1_verdict(-1, -0.1) == "이웃 그래프가 기여" and n1_verdict(-1, 1) == "구분 불가"
    assert r1_verdict(-1, 1) == "재현 안 됨(구분 불가)" and r1_verdict(-2, -1) == "재현됨"


def test_future_values_never_change_past_origin_predictions_none():
    with long_settings(), no_neighbors():
        base.test_future_values_never_change_past_origin_predictions()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
