"""대시보드 화면 꾸밈(design/ 디자인 키트). 계산은 하지 않는다.

색·글꼴·간격 값은 theme.css 하나에 있다: Streamlit 요소는 CSS로 칠하고, CSS 변수를 못 읽는 folium·Altair 는 TOK(같은 파일에서 읽은 값)로 칠한다.
위험 단계 색: 평상=정상(초록), 상승=주의(주황), 큰 상승·매우 큰 상승=위험(빨강, '매우 큰 상승'은 바깥 고리로 구분). 색만으로 구분하지 않게 항상 단계 이름을 같이 쓴다.
"""
import re
from html import escape
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st

from src.analysis.rise_levels import STAGE_MM, STAGE_NAMES

CSS = Path(__file__).with_name("theme.css").read_text(encoding="utf-8")


def _tokens(css: str) -> dict:
    root = re.search(r":root\s*\{(.*?)\n\}", css, re.S).group(1)
    raw = dict(re.findall(r"--([\w-]+):\s*([^;]+);", root))
    ref = lambda v: re.fullmatch(r"var\(--([\w-]+)\)", v.strip())
    while any(ref(v) for v in raw.values()):
        raw = {k: raw[ref(v).group(1)] if ref(v) else v.strip() for k, v in raw.items()}
    return raw


TOK = _tokens(CSS)
STAGE_TONE = ("ok", "warn", "high", "high")
STAGE_COLORS = tuple(TOK[f"risk-{t}"] for t in STAGE_TONE)
STAGE_TEXT = {"ok": TOK["green-text"], "warn": TOK["orange-text"], "high": TOK["red"]}   # 칩 글자 규칙
UNSTABLE = TOK["risk-none"]
FONT = "Inter, Noto Sans KR, sans-serif"
NOTO = "https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap"

ICON = {   # design/dashboard.html 에서 복사한 선 아이콘
    "wave": '<path d="M3 10c3 0 3 2 6 2s3-2 6-2 3 2 6 2M3 16c3 0 3 2 6 2s3-2 6-2 3 2 6 2"/>',
    "cloud": '<path d="M7 17.5h10a4 4 0 0 0 .6-8A5.5 5.5 0 0 0 7 9.6a4 4 0 0 0 0 7.9z"/>',
    "clock": '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l2.5 1.5"/>',
    "pin": '<path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>',
}
_svg = lambda k: f'<svg viewBox="0 0 24 24" aria-hidden="true">{ICON[k]}</svg>'


def apply() -> None:
    """st.html 은 <style>·<svg> 를 지우므로 markdown(unsafe_allow_html)로 넣는다. 한글 글꼴은 여기서 불러온다(config 는 URL 글꼴 하나만 받음)."""
    st.markdown(f"<style>@import url('{NOTO}');\n{CSS}</style>", unsafe_allow_html=True)


def header() -> None:
    st.markdown(f'<header class="ui-header"><div class="ui-logo"><span class="mark">{_svg("wave")}</span>강남구 하수관로 수위 현황</div></header>', unsafe_allow_html=True)


def page_head(hello: str, t: pd.Timestamp) -> None:
    st.markdown(f'<p class="ui-hello">{escape(hello)}</p><h1 class="ui-hero">{t.month}월 {t.day}일 {t:%H:%M}</h1>', unsafe_allow_html=True)


def stage_chip(k: int, text: str, ring: bool = False) -> str:
    return f'<span class="ui-chip ui-chip--{STAGE_TONE[k]}{" ui-chip--ring" if ring else ""}"><span class="dot" aria-hidden="true"></span>{escape(text)}</span>'


def legend(missing: str) -> None:
    """단계 칩 + 회색(기록 불안정) + 빈 원(자료 없음). missing: 빈 원의 뜻."""
    chips = []
    for k, name in enumerate(STAGE_NAMES):
        rng = f"{STAGE_MM[k - 1]} mm 이상" if k == 3 else (f"{STAGE_MM[0]} mm 미만" if k == 0 else f"{STAGE_MM[k - 1]}~{STAGE_MM[k]} mm")
        chips.append(stage_chip(k, f"{name} · {rng}", ring=k == 3))
    chips += ['<span class="sep" aria-hidden="true"></span>',
              '<span class="ui-chip ui-chip--gray"><span class="dot" aria-hidden="true"></span>기록 불안정(단계 표시 안 함)</span>',
              f'<span class="ui-chip ui-chip--gray ui-chip--hollow"><span class="dot" aria-hidden="true"></span>{escape(missing)}</span>']
    st.markdown(f'<div class="ui-legend" role="list" aria-label="평소 대비 상승 단계">{"".join(chips)}</div>', unsafe_allow_html=True)


def empty(icon: str, title: str, desc: str) -> None:
    st.markdown(f'<div class="ui-empty"><div class="icon">{_svg(icon)}</div><div class="title">{escape(title)}</div><div class="desc">{escape(desc)}</div></div>', unsafe_allow_html=True)


def skeleton(labels: list[str], cards: tuple[str, str]) -> None:
    """로딩: 제목·라벨은 진짜 글자, 값 자리만 회색 줄."""
    sk = lambda w, h, m="14px 0 0": f'<span class="ui-skel" style="width:{w};height:{h}px;margin:{m}"></span>'
    lines = "".join(sk(w, 13) for w in ("80%", "64%", "72%", "48%"))
    stats = "".join('<div class="ui-stat"><div class="label">' + escape(l) + "</div>" + sk("46px", 34, "8px 0 0") + "</div>" for l in labels)
    row = "".join('<div class="ui-card"><h2>' + escape(c) + "</h2>" + lines + "</div>" for c in cards)
    st.markdown(f'<div aria-busy="true" aria-label="자료를 읽는 중이에요">{sk("220px", 17, "18px 0 0")}{sk("360px", 48, "12px 0 0")}'
            f'<div class="ui-stats">{stats}</div><div class="ui-row">{row}</div></div>', unsafe_allow_html=True)


def sensor_marker(m: folium.Map, lat: float, lon: float, color: str | None, big: bool, selected: bool, ring: bool, tooltip: str) -> None:
    """수위 센서 원. color None = 자료 없음(빈 점선 원). 고른 센서는 검정 고리, '매우 큰 상승'은 빨강 바깥 고리."""
    r = 11 if big else 7
    if selected:
        folium.CircleMarker([lat, lon], radius=r + 7, color=TOK["ink"], weight=2, fill=False, interactive=False).add_to(m)
    if ring:
        folium.CircleMarker([lat, lon], radius=r + 3, color=TOK["risk-high"], weight=1.5, fill=False, interactive=False).add_to(m)
    folium.CircleMarker([lat, lon], radius=r, color="#fff" if color else TOK["ink-4"], weight=2 if color else 1.5, fill=True,
                        fill_color=color or "#fff", fill_opacity=0.95 if color else 0.0, dash_array=None if color else "3 3",
                        tooltip=tooltip).add_to(m)


def base_map(lat: float, lon: float) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    # 지도는 iframe 이라 theme.css 가 닿지 않는다: 바탕 지도만 회색·옅게(위험 색과 겹치지 않게). 옅은 타일(CartoDB)은 이제 API 키가 필요해 쓰지 않음
    m.get_root().header.add_child(folium.Element("<style>.leaflet-tile-pane{filter:grayscale(1) opacity(.55)}</style>"))
    return m


def stage_rules() -> alt.LayerChart:
    """단계 경계선(점선)과 이름. 경계 위 단계의 색."""
    df = pd.DataFrame({"y": list(STAGE_MM), "name": [f"{STAGE_NAMES[k + 1]} {mm}" for k, mm in enumerate(STAGE_MM)],
                       "line": list(STAGE_COLORS[1:]), "text": [STAGE_TEXT[t] for t in STAGE_TONE[1:]]})
    base = alt.Chart(df).encode(y="y:Q")
    return (base.mark_rule(strokeDash=[5, 4], strokeWidth=1.5).encode(color=alt.Color("line:N", scale=None))
            + base.mark_text(align="left", dy=-7, fontSize=11.5, fontWeight=600).encode(x=alt.value(6), text="name:N", color=alt.Color("text:N", scale=None)))


def chart(c, height: int) -> None:
    st.altair_chart(c.properties(height=height).configure(font=FONT)
                    .configure_axis(labelColor=TOK["ink-3"], titleColor=TOK["ink-3"], labelFontSize=11.5, titleFontSize=12, titleFontWeight=500,
                                    gridColor=TOK["line-3"], domainColor=TOK["line"], tickColor=TOK["line"])
                    .configure_view(stroke=None)
                    .configure_legend(labelColor=TOK["ink-2"], labelFontSize=13, symbolStrokeWidth=2),
                    width="stretch", theme=None)
