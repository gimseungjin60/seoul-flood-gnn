"""실행: python -m tests.test_lead_elev  (pytest로도 실행 가능). 먼저 오르는 센서-고도 규칙 함수 테스트."""
import numpy as np

from src.analysis.lead_elev_rules import e1_verdict, e2_verdict, elev_shuffle_null, leader_higher_frac, sensor_speed


def test_sensor_speed_removes_event_offset():
    ev = np.repeat([1, 2, 3, 4], 3)
    nd = np.tile(["a", "b", "c"], 4)
    delay = np.array([0, 10, 20, 100, 110, 120, -50, -40, -30, 5, 15, 25], float)
    s = sensor_speed(ev, nd, delay)
    assert s == {"a": -10.0, "b": 0.0, "c": 10.0}
    assert sensor_speed(ev[:9], nd[:9], delay[:9]) == {}            # 3사건뿐 → 제외


def test_leader_higher_frac_ignores_small_diff():
    elev = {"a": 30.0, "b": 20.0, "c": 21.0}
    p, n = leader_higher_frac([("a", "b"), ("b", "c"), ("c", "a")], elev)
    assert n == 2 and p == 0.5


def test_verdicts_and_null():
    assert e1_verdict(-0.6, -0.1) == "높은 센서가 먼저 오름"
    assert e1_verdict(-0.3, 0.2) == "고도와 반응 순서의 관계 구분 불가"
    elev = {k: float(i * 5) for i, k in enumerate("abcdefgh")}
    pairs = [(x, y) for x in "abcdefgh" for y in "abcdefgh" if elev[x] > elev[y]]
    null = elev_shuffle_null(pairs, elev, n_perm=300)
    assert e2_verdict(leader_higher_frac(pairs, elev)[0], null) == "앞선 센서가 더 높은 쪽"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
