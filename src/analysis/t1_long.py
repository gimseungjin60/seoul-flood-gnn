"""T1 확장(입력 120분, 30·60분 뒤) 실행: 규칙 src/analysis/t1_long_rules.py (docs/26). t1_st 와 같은 코드를 상수만 바꿔 실행한다.

실행: python -m src.analysis.t1_long          # 전체
      python -m src.analysis.t1_long --timing # GNN 1회 시간만
출력: data/processed/gangnam2026_t1_long_predictions.npz (재생성 가능, git 제외)
"""
import sys
from contextlib import contextmanager

from src.analysis import t1_st
from src.analysis.t1_long_rules import HORIZONS, WINDOW
from src.data.clean_nodes import PROCESSED

OUT_PRED = PROCESSED / "gangnam2026_t1_long_predictions.npz"


@contextmanager
def long_settings():
    old = (t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED)
    t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED = WINDOW, HORIZONS, OUT_PRED
    try:
        yield
    finally:
        t1_st.WINDOW, t1_st.HORIZONS, t1_st.OUT_PRED = old


def main() -> None:
    with long_settings():
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a == "--timing"]
        t1_st.main()


if __name__ == "__main__":
    main()
