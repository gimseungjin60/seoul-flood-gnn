"""이야기형 중간발표 덱 전용 그림 2장. 수치는 기존 결과 파일(gangnam_baseline_metrics_overall.csv,
gangnam_rain_baseline_report.txt)의 값만 쓴다.

실행: python -m src.analysis.story_figures
출력: outputs/figures/slide/{s04_sensor_map,s10_before_after}.png
- s04: 강남구 도로망 위에 분석에 쓴 센서 18개 위치 (도로망 © OpenStreetMap contributors는 슬라이드 각주에 표기)
- s10: 시간 예측(T1) 오차 전/후 비교. test 구간, 예측 10분 후, 단위 mm
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.slide_figures import FS, OUT, _save  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.visualization.make_figures import C  # noqa: E402

# gangnam_rain_baseline_report.txt [h=10분] test 행 (m -> mm): persistence / ridge_none
TEST_ALL = (7.3, 7.2)
TEST_EVENT = (18.0, 16.2)


def sensor_map():
    nodes = pd.read_csv(PROCESSED / "gangnam_road_nodes.csv", encoding="utf-8-sig").set_index("osmid")
    edges = pd.read_csv(PROCESSED / "gangnam_road_edges.csv", encoding="utf-8-sig")
    st = pd.read_csv(PROCESSED / "gangnam_station_graph_nodes.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.4, 8.2), dpi=130)
    ax.set_aspect(1 / 0.79)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    major = edges.highway.astype(str).str.contains("primary|secondary|trunk|tertiary")
    for sub, lw, col in ((edges[~major], 0.5, "#C9CED6"), (edges[major], 1.1, "#9AA5B5")):
        for u, v in zip(sub.u, sub.v):
            if u in nodes.index and v in nodes.index:
                ax.plot([nodes.x[u], nodes.x[v]], [nodes.y[u], nodes.y[v]], color=col, lw=lw, zorder=1, solid_capstyle="round")
    ax.scatter(st.lon, st.lat, s=210, color="#2A78D6", edgecolor="white", linewidth=2.2, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "s04_sensor_map")


def before_after():
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=130)
    groups = [("전체 구간", TEST_ALL), ("비 온 구간", TEST_EVENT)]
    w = 0.34
    for i, (name, (p, r)) in enumerate(groups):
        ax.bar(i - w / 2 - 0.01, p, w, color="#9AA5B5", label="직전 값 그대로" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, r, w, color="#2A78D6", label="선형 예측 모델" if i == 0 else None)
        ax.text(i - w / 2 - 0.01, p + 0.5, f"{p:.1f}", ha="center", fontsize=FS + 2, color=C["ink"], fontweight="bold")
        ax.text(i + w / 2 + 0.01, r + 0.5, f"{r:.1f}", ha="center", fontsize=FS + 2, color=C["ink"], fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels([g[0] for g in groups], fontsize=FS + 2)
    ax.set_ylabel("10분 뒤 수위 예측 오차 (mm)")
    ax.set_ylim(0, 22)
    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "s10_before_after")


if __name__ == "__main__":
    sensor_map()
    before_after()
