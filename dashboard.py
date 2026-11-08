"""강남구 하수관로 수위 대시보드 뼈대 (2026-06~08). 표시 기준: docs/16.

실행: streamlit run dashboard.py
- 지도: 고른 시각의 '평소 대비 상승 단계'. 위험 판정이 아니다. 기록 불안정 센서는 회색, 센서 없는 곳은 칠하지 않는다.
- 그래프: 고른 센서의 상승폭과 강우. 평가 대상 17개는 그 센서를 가리고 다른 센서로 추정한 값(IDW, GNN)을 비교선으로 함께 그린다.
"""
import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.analysis.rise_levels import STAGE_MM, STAGE_NAMES
from src.dashboard.data import load

STAGE_COLORS = ("#CFE0F1", "#8DBFE2", "#3A86C0", "#0B3D78")   # 물 색 계열(경보색을 쓰지 않음)
UNSTABLE = "#9AA5B5"

st.set_page_config(page_title="강남구 하수관로 수위", layout="wide")


@st.cache_data(show_spinner="자료를 읽는 중")
def data():
    return load()


D = data()
info, bt, events = D["info"], D["bt"], D["events"]

st.title("강남구 하수관로 수위: 평소 대비 상승")
st.caption("서울시 강남구 하수관로 수위 센서 48개, 2026-06-01 ~ 08-31. 상승폭 = 10분 평균 수위 − 그 센서의 요일·시간대별 평소 수위. "
           "관로 크기를 알 수 없어 **위험 판정이 아니라 평소보다 얼마나 올랐는지**만 보여줍니다.")

with st.sidebar:
    st.header("보기 설정")
    labels = [f"사건 {r.event_id}: {r.start:%m-%d %H:%M} ~ {r.end:%m-%d %H:%M} ({r.total_mm:.0f} mm)" for r in events.itertuples()]
    e = st.selectbox("비 사건", range(len(events)), format_func=lambda i: labels[i], index=int(events.total_mm.idxmax()))
    ev = events.iloc[e]
    win = np.flatnonzero((bt >= ev.start - pd.Timedelta(hours=3)) & (bt <= ev.end + pd.Timedelta(hours=3)))
    rows = D["rise"][win]
    peak = int(win[np.nanargmax(np.nanmax(np.where(D["stage"][win] >= 0, rows, -np.inf), axis=1))])
    t = st.select_slider("시각", options=list(bt[win]), value=bt[peak], format_func=lambda x: f"{x:%m-%d %H:%M}")
    ti = int(bt.get_loc(t))
    if "sensor" not in st.session_state:
        tg = np.flatnonzero((info.role == "평가 대상").to_numpy())            # 첫 화면: 이 사건에서 가장 많이 오른 평가 대상
        st.session_state.sensor = D["nodes"][tg[int(np.nanargmax(np.nanmax(rows[:, tg], axis=0)))]]
    st.selectbox("센서", D["nodes"], key="sensor",
                 format_func=lambda n: f"{n} ({info.set_index('node_id').loc[n, 'role']})")
    st.markdown("**단계 (평소 대비 상승)**")
    for k, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        rng = f"{STAGE_MM[k - 1]} mm 이상" if k == 3 else (f"{STAGE_MM[0]} mm 미만" if k == 0 else f"{STAGE_MM[k - 1]}~{STAGE_MM[k]} mm")
        st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background:{color};border:1px solid #556'></span> "
                    f"{name} ({rng})", unsafe_allow_html=True)
    st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background:{UNSTABLE}'></span> 기록 불안정(단계 표시 안 함)  \n"
                f"<span style='display:inline-block;width:12px;height:12px;border:1px dashed #556'></span> 이 시각 자료 없음(빈 원)",
                unsafe_allow_html=True)


def sensor_color(j: int) -> str | None:
    """None = 이 시각 자료 없음(빈 원)."""
    if info.unstable.iloc[j]:
        return UNSTABLE
    s = int(D["stage"][ti, j])
    return None if s < 0 else STAGE_COLORS[s]


left, right = st.columns([1.05, 1])
with left:
    st.subheader(f"{t:%m-%d %H:%M} 상승 단계")
    m = folium.Map(location=[info.lat.mean(), info.lon.mean()], zoom_start=13, tiles="OpenStreetMap")
    for j, r in info.iterrows():
        val = D["rise"][ti, j]
        txt = "자료 없음" if not np.isfinite(val) else f"{val:+.0f} mm"
        sel = r.node_id == st.session_state.sensor
        folium.CircleMarker([r.lat, r.lon], radius=11 if r.role == "평가 대상" else 7, color="#D1495B" if sel else "#33415C",
                            weight=3 if sel else 1.5, fill=True, fill_color=sensor_color(j) or "#FFFFFF",
                            fill_opacity=0.95 if sensor_color(j) else 0.0, dash_array=None if sensor_color(j) else "3 3",
                            tooltip=f"{r.node_id} · {r.role}{' · 기록 불안정' if r.unstable else ''} · {txt}").add_to(m)
    out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_object_clicked"])
    clicked = (out or {}).get("last_object_clicked")
    if clicked:
        d = (info.lat - clicked["lat"]) ** 2 + (info.lon - clicked["lng"]) ** 2
        n = info.node_id.iloc[int(d.idxmin())]
        if n != st.session_state.sensor:
            st.session_state.sensor = n
            st.rerun()
    st.caption("큰 원 = 평가 대상(연속 기록), 작은 원 = 이웃 센서. 센서가 없는 곳은 칠하지 않습니다.")

with right:
    n = st.session_state.sensor
    j = D["nodes"].index(n)
    r = info.iloc[j]
    st.subheader(f"{n} ({r.role}{', 기록 불안정' if r.unstable else ''})")
    ts = bt[win]
    df = pd.DataFrame({"시각": ts, "실측 상승폭": D["rise"][win, j]})
    lines = df.melt("시각", var_name="구분", value_name="mm")
    if n in D["est"]:
        e_df = D["est"][n].iloc[win].reset_index(names="시각").melt("시각", var_name="구분", value_name="mm")
        lines = pd.concat([lines, e_df[e_df["구분"] != "실측(같은 기준)"]])
    band = pd.DataFrame({"y0": [-1e4, *STAGE_MM], "y1": [*STAGE_MM, 1e4], "단계": list(STAGE_NAMES)})
    ymax = max(600, float(np.nanmax(lines.mm)) * 1.1 if lines.mm.notna().any() else 600)
    ymin = min(-50, float(np.nanmin(lines.mm)) * 1.1 if lines.mm.notna().any() else -50)
    rules = alt.Chart(pd.DataFrame({"y": list(STAGE_MM)})).mark_rule(color="#8A94A6", strokeDash=[3, 3]).encode(y="y:Q")
    now = alt.Chart(pd.DataFrame({"시각": [t]})).mark_rule(color="#D1495B").encode(x="시각:T")
    chart = alt.Chart(lines).mark_line().encode(
        x=alt.X("시각:T", title=None, axis=alt.Axis(format="%m-%d %H:%M")),
        y=alt.Y("mm:Q", title="평소 대비 상승 (mm)", scale=alt.Scale(domain=[ymin, ymax])),
        color=alt.Color("구분:N", scale=alt.Scale(domain=["실측 상승폭", "IDW 추정", "GNN 추정"], range=["#0B3D78", "#F08A24", "#2CA58D"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum["구분"] == "실측 상승폭", alt.value([1, 0]), alt.value([5, 3])))
    st.altair_chart((rules + chart + now).properties(height=300), width="stretch")
    rain = pd.DataFrame({"시각": ts, "강우 (mm/10분)": D["rain"][win, j]})
    st.altair_chart(alt.Chart(rain).mark_bar(color="#7FA7C9").encode(
        x=alt.X("시각:T", title=None, axis=alt.Axis(format="%m-%d %H:%M")), y="강우 (mm/10분):Q").properties(height=110),
        width="stretch")
    if n in D["est"]:
        st.info("점선은 이 센서를 가리고 다른 센서로 추정한 값입니다. 추정은 높은 단계를 실제보다 낮게 잡는 경향이 있습니다"
                "(실제 '매우 큰 상승'을 추정이 맞힌 비율 8~19%, docs/16). 추정선은 기준 수위 계산 방식이 지도와 조금 다릅니다.")
    else:
        st.caption("이웃 센서는 추정 비교선이 없습니다(평가 대상 17개만 가리고 추정해 봤습니다).")
