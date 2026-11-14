"""실행: python -m tests.test_t1_long  (pytest로도 실행 가능)
T1 확장(입력 120분, 30·60분 뒤) 누수 테스트: t1_st 의 누수 테스트를 확장 설정으로 다시 돌리고, 설정이 끝나면 원래 값으로 돌아오는지 본다."""
import numpy as np

from src.analysis import t1_st
from src.analysis.t1_long import OUT_PRED, long_settings
from src.analysis.t1_long_rules import HORIZONS, WINDOW
from tests import test_t1_st as base


def test_settings_applied_and_restored():
    before = (t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED)
    with long_settings():
        assert (t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED) == (WINDOW, HORIZONS, OUT_PRED)
        val, obs = t1_st.history(np.random.default_rng(0).normal(size=(40, 3)))
        assert val.shape == (40, 3, 12) and obs[:11, :, 0].sum() == 0
        y, _ = t1_st.targets(np.arange(40.0)[:, None].repeat(2, 1))
        assert np.allclose(y[5, :, 1], 6.0)                       # 60분 = 6블록
    assert (t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED) == before


def test_future_values_never_change_past_origin_predictions_long():
    with long_settings():
        base.test_future_values_never_change_past_origin_predictions()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
