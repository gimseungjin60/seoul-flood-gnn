"""2026 재비교 그림 2장. 수치는 결과 파일의 값만 쓴다(새로 계산하지 않는다).

실행: python -m src.analysis.figures_2026            # 둘 다 (모델 오차 그림은 gangnam2026_t2_ci.csv 가 있어야 한다)
출력: outputs/figures/slide/{s2026_sensor_map,s2026_gnn_ci,s2026_graph_diff}.png
- 지도: 2026 도로망(강남 범위) 위에 평가 대상(core, J2 통과)·이웃 후보·제외 센서 위치 (도로망 © OpenStreetMap contributors)
- 오차: 모델 5개(학습 없는 전체 평균·IDW 포함)의 비 온 구간 오차(이벤트 MAE)와 95% CI (t2_2026_eval 결과)
- 그래프 보완 실험: 사전 판정 Q1~Q3의 차이와 95% CI (t2_graph_eval 결과, gangnam2026_t2_graph_ci.csv 가 있을 때만)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from src.analysis.slide_figures import FS, _save  # noqa: E402
from src.data.build_station_graph_2026 import OUT_GRAPHML, OUT_NODES  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.visualization.make_figures import C  # noqa: E402

CI_CSV = PROCESSED / "gangnam2026_t2_ci.csv"
GRAPH_CI_CSV = PROCESSED / "gangnam2026_t2_graph_ci.csv"


def sensor_map():
    G = ox.load_graphml(OUT_GRAPHML)
    _, edges = ox.graph_to_gdfs(G)
    st = pd.read_csv(OUT_NODES)
    fig, ax = plt.subplots(figsize=(7.8, 7.4), dpi=130)
    ax.set_aspect(1 / 0.79)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    major = edges.highway.astype(str).str.contains("primary|secondary|trunk|tertiary")
    for sub, lw, col in ((edges[~major], 0.4, "#D3D8DF"), (edges[major], 1.0, "#9AA5B5")):
        ax.add_collection(LineCollection([list(g.coords) for g in sub.geometry], colors=col, linewidths=lw, zorder=1))
    groups = [(st.role == "neighbor", "이웃 후보 (주기적으로 끊기는 센서 포함)", "#F2A65A", 70),
              (st.role == "target", "평가 대상 (연속 기록 센서)", "#2A78D6", 150),
              (st.role == "dropped", "제외 (좌표 미확보·검증 실패·이상 점프)", "#9AA5B5", 70)]
    for m, lab, col, size in groups:
        d = st[m & st.lat.notna()]
        ax.scatter(d.lon, d.lat, s=size, color=col, edgecolor="white", linewidth=1.6, zorder=3,
                   label=f"{lab} {int(m.sum())}개", marker="o" if col != "#9AA5B5" else "X")
    pad = 0.006
    ax.set_xlim(st.lon.min() - pad, st.lon.max() + pad)
    ax.set_ylim(st.lat.min() - pad, st.lat.max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=FS - 2)
    fig.tight_layout()
    _save(fig, "s2026_sensor_map")


def gnn_ci():
    d = pd.read_csv(CI_CSV).set_index("stat")
    order = [("mean_others", "전체 평균"), ("idw_p1", "IDW"), ("ridge_rain_time_nbr", "강우+이웃\n(선형)"),
             ("hgb_rain_time_nbr", "강우+이웃\n(트리)"), ("gnn", "GNN")]
    vals = [tuple(d.loc[f"mae_{m}", c] * 1000 for c in ("point_m", "lo_m", "hi_m")) for m, _ in order]
    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=130)
    for i, ((m, _), (v, lo, hi)) in enumerate(zip(order, vals)):
        ax.bar(i, v, 0.58, color="#8798B3" if m in ("mean_others", "idw_p1") else "#2A78D6", zorder=3)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=C["ink"], elinewidth=2, capsize=6, zorder=4)
        ax.text(i, hi + max(hi for _, _, hi in vals) * 0.02, f"{v:.1f}", ha="center", fontsize=FS + 1, fontweight="bold", color=C["ink"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab for _, lab in order], fontsize=FS)
    ax.set_ylabel("비 온 구간 오차 (mm)")
    ax.set_ylim(0, max(hi for _, _, hi in vals) * 1.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "s2026_gnn_ci")


def graph_diff():
    d = pd.read_csv(GRAPH_CI_CSV).set_index("stat")
    rows = [("Q1_elev_minus_road", "고도 방향 GNN\n− 도로거리 GNN"), ("Q2_elev_minus_idw", "고도 방향 GNN\n− IDW"),
            ("Q3_road_minus_rand", "도로거리 GNN\n− 랜덤 GNN")]
    fig, ax = plt.subplots(figsize=(10.4, 4.6), dpi=130)
    for i, (k, _) in enumerate(rows):
        v, lo, hi = (d.loc[k, c] * 1000 for c in ("point_m", "lo_m", "hi_m"))
        col = "#2A78D6" if lo <= 0 <= hi else "#E8692E"
        ax.plot([lo, hi], [i, i], color=col, lw=5, solid_capstyle="round", zorder=3)
        ax.scatter([v], [i], s=160, color=col, edgecolor="white", linewidth=2, zorder=4)
        ax.text(hi + 0.25, i, f"{v:+.2f}", va="center", fontsize=FS + 1, fontweight="bold", color=C["ink"])
    ax.axvline(0, color=C["ink"], ls="--", lw=1.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlim(-2, 7)
    ax.set_xlabel("비 온 구간 오차 차이 (mm) · 선은 95% 범위 · 0보다 오른쪽이면 앞의 것이 오차가 큼")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "s2026_graph_diff")


if __name__ == "__main__":
    sensor_map()
    if CI_CSV.exists() or "--ci" in sys.argv:
        gnn_ci()
    if GRAPH_CI_CSV.exists():
        graph_diff()
