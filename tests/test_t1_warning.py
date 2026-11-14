"""실행: python -m tests.test_t1_warning  (pytest로도 실행 가능). 큰 상승 첫 도달 감지 규칙 함수 테스트."""
import numpy as np

from src.analysis.t1_warning_rules import case_outcome, csi, w_verdict


def test_reach_hit_and_miss():
    act = np.array([0, 50, 120, 250, 300, 100], float)
    issue = np.array([0, 0, 0, 50, 120, 250], float)          # 30분 전 실제
    assert case_outcome(act, np.array([0, 0, 0, 210, 0, 0.]), issue) == ("reach", True)
    assert case_outcome(act, np.array([0, 0, 0, 150, 400, 0.]), issue) == ("reach", False)
    assert case_outcome(act, act.copy(), issue) == ("reach", True)
    assert case_outcome(act, issue, issue) == ("reach", False)  # 직전 값 그대로는 정의상 놓침


def test_none_case_false_alarm_and_exclusions():
    act = np.array([0, 50, 120, 150, 100, 20], float)
    assert case_outcome(act, act, act) == ("none", False)
    assert case_outcome(act, np.array([0, 0, 0, 0, 0, 201.]), act) == ("none", True)
    assert case_outcome(act[:5], act[:5], act[:5]) is None                   # 유효 블록 6개 미만
    hi = np.array([250, 260, 270, 100, 50, 0], float)
    assert case_outcome(hi, hi, hi) is None                                  # 창 첫 블록부터 도달
    late = np.array([0, 50, 120, 150, 250, 0], float)
    assert case_outcome(late, late, np.array([0, 0, 0, 0, 210, 0.])) is None  # 기점에 이미 도달


def test_csi_and_verdict():
    assert csi(3, 5, 1) == 0.5
    assert np.isnan(csi(0, 0, 0))
    assert w_verdict(0.01, 0.2) == "GNN이 큰 상승 첫 도달을 더 잘 잡음"
    assert w_verdict(-0.1, 0.1) == "구분 불가"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
