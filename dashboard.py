"""강남구 하수관로 수위 현황. 표시 기준: docs/16, 강우 요약: docs/34.

실행: streamlit run dashboard.py   (실시간 모드는 .env 의 SEOUL_OPENAPI_KEY 필요, src/dashboard/live.py)
- 모드 2개: **지금(실시간)** = 열린데이터광장 최신 수위·강우를 평소와 비교(src/dashboard/live_view.py), **재생** = 2026-06~08 기록.
- 재생 모드: 비 사건과 시각을 골라 그때의 센서 상태를 본다. 실시간이 아니다.
- 위: 이 시각까지의 강우, 평소 대비 상승 단계별 센서 수, 과거 비슷한 강우에서의 요약(예측이 아니라 과거 사례).
- 지도: 센서별 평소 대비 상승 단계(위험 판정 아님). 기록 불안정 센서는 회색, 센서 없는 곳은 칠하지 않는다.
- 그래프: 고른 센서의 상승폭, 강우, 30분 전에 낸 예측(강우+자기이력 선형, docs/19).
- 화면 꾸밈(색·카드·배너·로딩/빈/오류 상태)은 src/dashboard/ui.py·theme.css(design/ 디자인 키트)에 있다.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.dashboard import live_view, response_view, ui
from src.dashboard.data import load
from src.dashboard.ui import STAGE_COLORS, TOK, UNSTABLE

HEAVY_MM = 50                                                   # docs/34: 총 50 mm 이상 사건 5개와 비교

st.set_page_config(page_title="강남구 하수관로 수위", layout="wide")
ui.apply()
ui.header()


@st.cache_data(show_spinner=False)
def data():
    return load()


loading = st.empty()
with loading.container():
    ui.skeleton(["이 사건 누적 강우", "지금 10분 강우", "'상승' 이상 센서", "'큰 상승' 이상 센서"], ("상승 단계", "센서"))
try:
    D = data()
except Exception as err:          # 화면에는 무엇이 안 됐는지와 할 일을, 원래 오류는 펼쳐 보기에
    loading.empty()
    st.warning(f"**대시보드 자료를 읽지 못했어요** · {err}\n\n원본·처리 결과 파일은 바뀌지 않았어요. "
               "`data/processed` 파일이 있는지 확인한 뒤 새로고침해 주세요(README).", icon=":material/warning:")
    with st.expander("자세한 오류"):
        st.exception(err)
    st.stop()
loading.empty()
info, bt, events, hist = D["info"], D["bt"], D["events"], D["history"]

with st.sidebar:
    mode = st.radio("모드", ["지금(실시간)", "재생(과거 사건)"], help="지금: 열린데이터광장 최신 자료 / 재생: 2026-06~08 기록")
if mode.startswith("지금"):
    live_view.render(D, STAGE_COLORS, UNSTABLE)
    st.stop()

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

# ---------------------------------------------------------------- 요약 (이 시각)
stable = ~info.unstable.to_numpy()
st_now = D["stage"][ti]
valid = stable & (st_now >= 0)
n_valid = int(valid.sum())
n_up, n_big, n_vbig = (int((valid & (st_now >= k)).sum()) for k in (1, 2, 3))
in_ev = (bt >= ev.start) & (bt <= t)
cum_mm = float(np.nansum(D["rain_gn"][in_ev]))
now_mm = float(D["rain_gn"][ti]) if np.isfinite(D["rain_gn"][ti]) else 0.0

ui.page_head(f"재생 모드 · 사건 {ev.event_id} 다시 보기", t)
st.info("**재생 모드**: 2026년 6~8월의 실제 기록을 재생합니다. 지금의 실시간 수위가 아닙니다.", icon=":material/history:")

c1, c2, c3, c4 = st.columns(4)
c1.metric("이 사건 누적 강우", f"{cum_mm:.0f} mm", help="강남 강우량계 평균, 사건 시작부터 이 시각까지")
c2.metric("지금 10분 강우", f"{now_mm:.1f} mm")
c3.metric("'상승' 이상 센서", f"{n_up} / {n_valid}", help="평소보다 50 mm 이상 오른 센서 수(기록 안정 센서만)")
c4.metric("'큰 상승' 이상 센서", f"{n_big} / {n_valid}", help="평소보다 200 mm 이상 오른 센서 수")

heavy, light = hist[hist.total_mm >= HEAVY_MM], hist[hist.total_mm < 30]
if cum_mm >= HEAVY_MM:
    g, name = heavy, f"{HEAVY_MM} mm 이상"
elif cum_mm < 30:
    g, name = light, "30 mm 미만"
else:
    g = None
if g is not None:
    st.info(f"**과거 사례 요약(예측 아님)** · 지금까지 {cum_mm:.0f} mm가 내렸습니다. 올해 6~8월 사건 중 최종 총강우량이 {name}인 사건 {len(g)}건에서는 센서 약 {int(g.n_sensors.median())}개 중 "
            f"'큰 상승' 이상이 **{int(g.n_big.min())}~{int(g.n_big.max())}개**, '매우 큰 상승'이 {int(g.n_vbig.min())}~{int(g.n_vbig.max())}개까지 올랐습니다. "
            "비가 더 오면 총량이 늘어 해당 구간이 바뀝니다. 총강우량과 센서 상승은 함께 움직였지만(순위상관 0.90, 사건 13개) 같은 강우라도 편차가 큽니다. 위험 판정이 아닙니다."
            .replace("~", r"\~"), icon=":material/info:")   # 마크다운이 '~ … ~'를 취소선으로 읽지 않게
else:
    st.info(f"지금까지 {cum_mm:.0f} mm로 30~50 mm 사이입니다. 올해 6~8월에는 최종 총강우량이 이 구간인 사건이 없어 비교할 과거 사례가 없습니다.",
            icon=":material/info:")


def sensor_color(j: int) -> str | None:
    """None = 이 시각 자료 없음(빈 원)."""
    if info.unstable.iloc[j]:
        return UNSTABLE
    s = int(D["stage"][ti, j])
    return None if s < 0 else STAGE_COLORS[s]


left, right = st.columns([1.05, 1], gap="medium")
with left, st.container(key="card_map"):
    st.subheader(f"{t:%m-%d %H:%M} 상승 단계")
    m = ui.base_map(info.lat.mean(), info.lon.mean())
    for j, r in info.iterrows():
        val = D["rise"][ti, j]
        txt = "자료 없음" if not np.isfinite(val) else f"{val:+.0f} mm"
        ui.sensor_marker(m, r.lat, r.lon, sensor_color(j), big=r.role == "평가 대상", selected=r.node_id == st.session_state.sensor,
                         ring=not r.unstable and D["stage"][ti, j] == 3,
                         tooltip=f"{r.node_id} · {r.role}{' · 기록 불안정' if r.unstable else ''} · {txt}{r.flood_txt}")
    out = st_folium(m, height=480, use_container_width=True, returned_objects=["last_object_clicked"])
    clicked = (out or {}).get("last_object_clicked")
    if clicked:
        d = (info.lat - clicked["lat"]) ** 2 + (info.lon - clicked["lng"]) ** 2
        n = info.node_id.iloc[int(d.idxmin())]
        if n != st.session_state.sensor:
            st.session_state.sensor = n
            st.rerun()
    ui.legend("이 시각 자료 없음(빈 원)")
    st.caption("큰 원 = 평가 대상(연속 기록), 작은 원 = 이웃 센서. 센서가 없는 곳은 칠하지 않습니다(추정 근거가 약합니다, docs/15·16).")

with right, st.container(key="card_sensor"):
    n = st.session_state.sensor
    j = D["nodes"].index(n)
    r = info.iloc[j]
    st.subheader(f"{n} ({r.role}{', 기록 불안정' if r.unstable else ''})")
    ts = bt[win]
    lines = pd.DataFrame({"시각": ts, "구분": "실측 상승폭", "mm": D["rise"][win, j]})
    if n in D["fc"]:
        f_df = D["fc"][n].iloc[win].rename("mm").rename_axis("시각").reset_index().assign(구분="30분 전 예측(선형)")
        lines = pd.concat([lines, f_df[["시각", "구분", "mm"]]])
    ymax = max(600, float(np.nanmax(lines.mm)) * 1.1 if lines.mm.notna().any() else 600)
    ymin = min(-50, float(np.nanmin(lines.mm)) * 1.1 if lines.mm.notna().any() else -50)
    rules = ui.stage_rules()
    now = alt.Chart(pd.DataFrame({"시각": [t]})).mark_rule(color=TOK["red-live"], strokeWidth=2).encode(x="시각:T")
    chart = alt.Chart(lines).mark_line().encode(
        x=alt.X("시각:T", title=None, axis=alt.Axis(format="%m-%d %H:%M")),
        y=alt.Y("mm:Q", title="평소 대비 상승 (mm)", scale=alt.Scale(domain=[ymin, ymax])),
        color=alt.Color("구분:N", scale=alt.Scale(domain=["실측 상승폭", "30분 전 예측(선형)"], range=[TOK["ink"], TOK["estimate"]]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum["구분"] == "실측 상승폭", alt.value([1, 0]), alt.value([5, 3])))
    ui.chart(rules + chart + now, 280)
    rain = pd.DataFrame({"시각": ts, "강우 (mm/10분)": D["rain"][win, j]})
    ui.chart(alt.Chart(rain).mark_bar(color=TOK["rain-bar"]).encode(
        x=alt.X("시각:T", title=None, axis=alt.Axis(format="%m-%d %H:%M")), y="강우 (mm/10분):Q"), 100)
    if n in D["fc"]:
        st.caption("파랑 점선: 그 시각의 값을 30분 앞서 강우와 자기 수위 이력으로 예측한 것. 직전 값 그대로와 통계적으로 구분되지 않았고 "
                   "큰 상승은 낮게 잡는 경향이 있습니다(docs/19).")
    else:
        st.caption("이웃 센서는 예측선이 없습니다(평가 대상 17개만 예측했습니다).")
with right:
    response_view.render(n, int(ev.event_id))
