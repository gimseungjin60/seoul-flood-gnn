"""먼저 오르는 센서가 더 높은 곳에 있는가: 사전 고정 규칙(src/analysis/lead_elev_rules.py, docs/22)대로 E1·E2 판정과 참고 집계.

실행: python -m src.analysis.lead_elev   (먼저 python -m src.analysis.lead_lag 결과가 있어야 한다)
출력: data/processed/gangnam2026_lead_elev_report.txt, gangnam2026_lead_elev_sensors.csv
"""
import numpy as np
import pandas as pd

from src.analysis.lead_elev_rules import (ELEV_DIFF_MIN, FREEZE_DATE, e1_verdict, e2_verdict, elev_shuffle_null,
                                          leader_higher_frac, rho_ci, sensor_speed)
from src.analysis.lead_lag import OUT_EVTAB, OUT_PAIRS
from src.data.clean_nodes import PROCESSED
from src.data.sensor_elevation_2026 import OUT_ELEV

OUT_REPORT = PROCESSED / "gangnam2026_lead_elev_report.txt"
OUT_SENS = PROCESSED / "gangnam2026_lead_elev_sensors.csv"


def main() -> None:
    ev = pd.read_csv(OUT_EVTAB)
    v = ev[ev.valid]
    speed = sensor_speed(v.event_id, v.node_id, v.delay)
    elev = pd.read_csv(OUT_ELEV).set_index("node_id")["elev_m"].to_dict()
    sens = pd.DataFrame({"speed_min": speed}).join(pd.Series(elev, name="elev_m"))
    sens.index.name = "node_id"
    sens.sort_values("speed_min").to_csv(OUT_SENS, encoding="utf-8-sig", float_format="%.2f")

    def e1(s):
        return rho_ci(s.speed_min.to_numpy(), s.elev_m.to_numpy())

    rho, lo, hi = e1(sens)
    pt = pd.read_csv(OUT_PAIRS)
    c = pt[pt.consistent]
    pairs = [(r.leader, r.node_j if r.leader == r.node_i else r.node_i) for r in c.itertuples()]
    p, n2 = leader_higher_frac(pairs, elev)
    null = elev_shuffle_null(pairs, elev)
    road = [(r.leader, r.node_j if r.leader == r.node_i else r.node_i) for r in c[c.road_nbr].itertuples()]
    pr, nr = leader_higher_frac(road, elev)
    r17 = e1(sens.drop("23-0017", errors="ignore"))
    r89 = e1(sens.drop("23-0089", errors="ignore"))

    L = [f"먼저 오르는 센서와 고도 (규칙 고정 {FREEZE_DATE}, docs/22). 서울시 강남구 2026, 반응 속도를 정한 센서 {len(sens)}개",
         "",
         "[판정]",
         f"  E1 (주) {e1_verdict(lo, hi)}: 반응 속도(작을수록 먼저)와 고도의 순위 상관 {rho:.3f}, 95% 구간 [{lo:.3f}, {hi:.3f}]",
         f"  E2 (보조) {e2_verdict(p, null)}: 일관된 시차 쌍 중 고도 차 ≥ {ELEV_DIFF_MIN:.0f} m 인 {n2}쌍에서 앞선 센서가 높은 비율 "
         f"{p:.3f} (고도 섞기 우연 2.5~97.5백분위 [{np.nanpercentile(null, 2.5):.3f}, {np.nanpercentile(null, 97.5):.3f}], "
         f"평균 {np.nanmean(null):.3f})",
         "",
         "[참고 — 판정에 쓰지 않음]",
         f"  도로 이웃의 일관된 쌍만: 고도 차 ≥ {ELEV_DIFF_MIN:.0f} m 인 {nr}쌍 중 앞선 센서가 높은 비율 {pr:.3f}",
         f"  23-0017 제외 E1: {r17[0]:.3f} [{r17[1]:.3f}, {r17[2]:.3f}]",
         f"  23-0089 제외(23-0088과 같은 위치) E1: {r89[0]:.3f} [{r89[1]:.3f}, {r89[2]:.3f}]",
         "",
         "[센서별] 반응 속도(분, 작을수록 먼저) / 고도(m)"] + \
        [f"  {n}  {r.speed_min:+.0f}  {r.elev_m:.1f}" for n, r in sens.sort_values("speed_min").iterrows()] + \
        ["", "[한계] 30 m 표면 고도는 관로 깊이·경사가 아님. 강남은 대부분 평지. 관로가 지형을 따르지 않을 수 있음."]
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
