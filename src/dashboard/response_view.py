"""대시보드 센서별 '과거 강우 반응' 패널. 자료는 src/analysis/rain_response.py 결과(CSV)만 읽는다(새로 계산하지 않음, docs/21).
CSV 가 없으면 아무것도 그리지 않는다. 예측이 아니라 과거 사건에서 이 센서가 얼마나 올랐는지의 요약이다."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.rise_levels import STAGE_MM, STAGE_NAMES
from src.analysis.rain_response import OUT_PAIRS, OUT_SENSOR


@st.cache_data
def _tables():
    if not (OUT_SENSOR.exists() and OUT_PAIRS.exists()):
        return None
    return pd.read_csv(OUT_SENSOR).set_index("node_id"), pd.read_csv(OUT_PAIRS)


def render(node_id: str, highlight_event: int | None = None) -> None:
    t = _tables()
    if t is None:
        return
    s, p = t
    st.markdown("**이 센서의 과거 강우 반응** (2026-06~08, 사건별 최대 상승)")
    if node_id not in s.index:
        st.caption("기록이 불안정해 요약하지 않았습니다.")
        return
    r = s.loc[node_id]
    g = p[p.node_id == node_id].copy()
    g["이 사건"] = g.event_id == highlight_event
    pts = alt.Chart(g).mark_circle(size=90, opacity=0.85).encode(
        x=alt.X("total_mm:Q", title="사건 총강우 (mm)"), y=alt.Y("peak_mm:Q", title="최대 상승 (mm)"),
        color=alt.condition(alt.datum["이 사건"], alt.value("#D1495B"), alt.value("#0B3D78")),
        tooltip=[alt.Tooltip("event_id:Q", title="사건"), alt.Tooltip("total_mm:Q", title="총강우 mm", format=".0f"),
                 alt.Tooltip("peak_mm:Q", title="최대 상승 mm", format=".0f")])
    rules = alt.Chart(pd.DataFrame({"y": list(STAGE_MM)})).mark_rule(color="#8A94A6", strokeDash=[3, 3]).encode(y="y:Q")
    st.altair_chart((rules + pts).properties(height=200), width="stretch")
    fmt = lambda pre: (f"{r[pre + '_median_mm']:.0f} mm (범위 {r[pre + '_min_mm']:.0f}~{r[pre + '_max_mm']:.0f}, {int(r[pre + '_n'])}건)"
                       if r[pre + "_n"] else "사건 없음")
    st.caption(f"총 30 mm 미만 비: 최대 상승 중앙 {fmt('light')} · 50 mm 이상 비: {fmt('heavy')} · 사건 13개 중 최고 {STAGE_NAMES[int(r.max_stage_reached)]}")
    if r.verdict == "강우와 함께 움직임":
        st.caption(f"강우가 많을수록 더 올랐습니다(순위상관 {r.rho:+.2f}, 95% 구간 {r.rho_lo:+.2f}~{r.rho_hi:+.2f}). 예측이 아니라 과거 사례입니다.")
    else:
        st.caption("이 자료(사건 13개)로는 강우와의 관계를 확인하지 못했습니다. 반응이 없다는 뜻이 아닙니다.")
