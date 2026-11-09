"""대시보드 '지금(실시간)' 화면. 자료·계산은 src/dashboard/live.py, 단계 기준은 docs/16. 예측이 아니라 **최신 관측**을 평소와 비교해 보여준다."""
import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.analysis.rise_levels import STAGE_MM, STAGE_NAMES, stage
from src.dashboard import live, response_view


@st.cache_data(ttl=300, show_spinner="열린데이터광장에서 최신 자료를 읽는 중")
def _fetch():
    now = live.now_kst()
    return now, live.fetch_sewer(now), live.fetch_rain()


def render(D: dict, colors: tuple, unstable_color: str) -> None:
    st.title("강남구 하수관로 수위 현황 (지금)")
    try:
        now, sewer, rain = _fetch()
    except live.LiveError as e:
        st.error(f"실시간 자료를 읽지 못했습니다: {e}")
        st.caption("`재생(과거 사건)` 모드는 그대로 쓸 수 있습니다. 인증키는 `.env`의 SEOUL_OPENAPI_KEY 입니다.")
        return
    nodes, info = D["nodes"], D["info"]
    lr = live.latest_rise(sewer, nodes, D["prof"], now)
    lr["stage"] = np.where(lr.ok, stage(np.nan_to_num(lr.rise_mm)), -1)
    stable = ~info.unstable.to_numpy()
    valid = stable & lr.ok.to_numpy()
    n_valid = int(valid.sum())
    n_up, n_big = (int((valid & (lr.stage.to_numpy() >= k)).sum()) for k in (1, 2))
    last_t = lr.time[lr.ok].max()
    r1h = rain[rain.time > rain.time.max() - pd.Timedelta(minutes=60)].groupby("gauge").mm.sum()

    top = st.columns([1, 1, 1, 1, 0.6])
    top[0].metric("가장 최근 측정", f"{last_t:%H:%M}", help=f"지금 {now:%H:%M}. 자료 지연 약 {(now - last_t).total_seconds() / 60:.0f}분")
    top[1].metric("최근 1시간 강우", f"{r1h.mean():.1f} mm" if len(r1h) else "자료 없음", help="강남구 강우량계 평균(10분 자료 합)")
    top[2].metric("'상승' 이상 센서", f"{n_up} / {n_valid}", help="평소보다 50 mm 이상 높은 센서(기록 안정·최근 측정 센서만)")
    top[3].metric("'큰 상승' 이상 센서", f"{n_big} / {n_valid}", help="평소보다 200 mm 이상 높은 센서")
    if top[4].button("새로고침"):
        st.cache_data.clear()
        st.rerun()

    if len(r1h) and r1h.mean() == 0:
        st.info("최근 1시간 강남구에 비가 오지 않았습니다.")
    st.warning("**평소 수위 기준이 2026년 6~8월(여름) 자료입니다.** 계절이 달라 비가 없어도 평소보다 높거나 낮게 나올 수 있습니다. "
               "위험 판정이 아니라 평소 대비 값이며, 자료는 열린데이터광장 OpenAPI라 약 10~30분 늦습니다.", icon="⚠️")

    sel = st.session_state.get("live_sensor", nodes[0])
    left, right = st.columns([1.05, 1])
    with left:
        m = folium.Map(location=[info.lat.mean(), info.lon.mean()], zoom_start=13, tiles="OpenStreetMap")
        for j, r in info.iterrows():
            row = lr.iloc[j]
            col = unstable_color if r.unstable else (colors[int(row.stage)] if row.ok else None)
            txt = f"{row.rise_mm:+.0f} mm · {row.age_min:.0f}분 전" if row.ok else f"자료 없음({row.status})"
            folium.CircleMarker([r.lat, r.lon], radius=11 if r.role == "평가 대상" else 7, color="#D1495B" if r.node_id == sel else "#33415C",
                                weight=3 if r.node_id == sel else 1.5, fill=True, fill_color=col or "#FFFFFF",
                                fill_opacity=0.95 if col else 0.0, dash_array=None if col else "3 3",
                                tooltip=f"{r.node_id} · {r.role}{' · 기록 불안정' if r.unstable else ''} · {txt}{r.flood_txt}").add_to(m)
        out = st_folium(m, height=480, use_container_width=True, returned_objects=["last_object_clicked"], key="live_map")
        c = (out or {}).get("last_object_clicked")
        if c:
            n = info.node_id.iloc[int(((info.lat - c["lat"]) ** 2 + (info.lon - c["lng"]) ** 2).idxmin())]
            if n != sel:
                st.session_state.live_sensor = n
                st.rerun()
        st.caption("색은 평소 대비 상승 단계(회색=기록 불안정, 빈 원=최근 측정 없음·통신 이상·오류값). 센서가 없는 곳은 칠하지 않습니다.")
        for k, (name, col) in enumerate(zip(STAGE_NAMES, colors)):
            rng = f"{STAGE_MM[k - 1]} mm 이상" if k == 3 else (f"{STAGE_MM[0]} mm 미만" if k == 0 else f"{STAGE_MM[k - 1]}~{STAGE_MM[k]} mm")
            st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background:{col};border:1px solid #556'></span> {name} ({rng})",
                        unsafe_allow_html=True)
    with right:
        j = nodes.index(sel)
        st.subheader(f"{sel} ({info.role.iloc[j]}{', 기록 불안정' if info.unstable.iloc[j] else ''})")
        ser = live.rise_series(sewer, sel, j, D["prof"])
        if ser.empty:
            st.caption("이 센서의 최근 측정이 없습니다.")
        else:
            rules = alt.Chart(pd.DataFrame({"y": list(STAGE_MM)})).mark_rule(color="#8A94A6", strokeDash=[3, 3]).encode(y="y:Q")
            line = alt.Chart(ser).mark_line(point=True, color="#0B3D78").encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("rise_mm:Q", title="평소 대비 상승 (mm)", scale=alt.Scale(domain=[min(-50, float(ser.rise_mm.min()) * 1.1),
                                                                                         max(600, float(ser.rise_mm.max()) * 1.1)])))
            st.altair_chart((rules + line).properties(height=280), width="stretch")
        rg = rain.groupby("time").mm.mean().reset_index()
        st.altair_chart(alt.Chart(rg).mark_bar(color="#7FA7C9").encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")), y=alt.Y("mm:Q", title="강남 강우 (mm/10분)")).properties(height=100),
            width="stretch")
        st.caption("센서 선택은 지도에서 원을 클릭하세요. 이웃 센서 일부는 약 2시간 주기로만 기록해 점이 드뭅니다.")
        response_view.render(sel)
