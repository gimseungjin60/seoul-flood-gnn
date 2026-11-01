"""실행: python -m tests.test_t2_2026_rules  (pytest로도 실행 가능)"""
import numpy as np
import pandas as pd

import src.analysis.t2_2026_rules as r


def test_verdicts():
    assert r.p1_verdict(-2.0, -0.1) == "GNN이 IDW를 넘음"
    assert r.p1_verdict(-0.3, 2.1) == "구분 불가"
    assert r.p1_verdict(0.1, 2.0) == "IDW보다 나쁨"
    assert r.p2_graph_effect("GNN이 IDW를 넘음", -0.01) and not r.p2_graph_effect("구분 불가", -1.0)
    assert not r.p2_graph_effect("GNN이 IDW를 넘음", 0.0)
    assert r.p3_verdict(-30, -1) == "첨두 시각에서 GNN이 IDW를 넘음" and r.p3_verdict(-5, 5) == "구분 불가"


def test_p3_eligibility():
    assert r.p3_eligible(np.array([0.0, 0.05, np.nan])) and not r.p3_eligible(np.array([0.01, 0.049]))


def test_s1_matches_2025_ratio():
    assert r.s1_min_sensors(18) == 6 and r.s1_min_sensors(20) == 7 and r.s1_min_sensors(17) == 6
    assert r.s1_verdict(7, 20) and not r.s1_verdict(6, 20)


def test_eval_events_drops_truncated():
    ev = pd.DataFrame({"event_id": [0, 1], "start": pd.to_datetime(["2026-08-22 05:10", "2026-08-31 01:10"]),
                       "end": pd.to_datetime(["2026-08-22 17:49", "2026-08-31 23:59"])})
    out = r.eval_events(ev, pd.Timestamp("2026-08-31 23:59"))
    assert out.event_id.tolist() == [0]


def test_jump_exclusion_and_roles():
    v = np.full((300, 3), 0.1)
    v[::2, 1] = 1.5                     # 센서 B: 1분마다 1.4 m 왕복 → 점프 299회
    v[10, 2] = 1.2                      # 센서 C: 점프 2회
    v[50:60, 2] = np.nan                # NaN 사이는 세지 않음
    assert r.jump_excluded(v, ["A", "B", "C"]) == ["B"]
    sel = pd.DataFrame({"node_id": ["23-0001", "23-0039", "23-0052", "23-0099", "23-0100"],
                        "is_core": [True, True, False, False, True],
                        "in_pool": [True, True, True, True, True]})
    roles = r.sensor_roles(sel, ["23-0052"], ["23-0100"])
    assert roles["targets"] == ["23-0001", "23-0039"] and roles["H"] == ["23-0039"]
    assert roles["pool"] == ["23-0001", "23-0039", "23-0099"] and roles["dropped"] == ["23-0052", "23-0100"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
