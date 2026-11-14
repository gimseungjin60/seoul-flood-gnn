"""실행: python -m tests.test_t1_adapt  (pytest로도 실행 가능)
G-adapt 테스트: 미래 값 누수 없음(t1_st 누수 테스트를 G-adapt 로 다시), 관측 안 된 이웃은 메시지에 섞이지 않음, 모델 교체가 원래대로 돌아옴."""
import torch

from src.analysis import t1_st
from src.analysis.t1_adapt import AdaptiveSTGNN, adaptive_model
from tests import test_t1_st as base


def test_model_swap_restored():
    old = t1_st.STGNN
    with adaptive_model(base.N):
        assert t1_st.STGNN is AdaptiveSTGNN
    assert t1_st.STGNN is old


def test_unobserved_neighbor_has_no_effect():
    torch.manual_seed(0)
    with adaptive_model(5):
        model = AdaptiveSTGNN(3).eval()
    seq, stat = torch.randn(2, 5, 6, 2), torch.randn(2, 5, 3)
    m = torch.ones(2, 5)
    m[:, 4] = 0
    seq2 = seq.clone()
    seq2[:, 4] = torch.randn(2, 6, 2) * 10                    # 관측 안 된 센서 4의 값만 바꿈
    with torch.no_grad():
        a, b = model(seq, stat, m), model(seq2, stat, m)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)
    m0 = torch.zeros(2, 5)                                     # 이웃이 하나도 관측 안 됨 → NaN 없음
    with torch.no_grad():
        assert torch.isfinite(model(seq, stat, m0)).all()


def test_future_values_never_change_past_origin_predictions_adapt():
    with adaptive_model(base.N):
        base.test_future_values_never_change_past_origin_predictions()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
