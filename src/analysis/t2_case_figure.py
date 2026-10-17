"""사례 그림: 23-0014는 가장 가까운 이웃(23-0015)이 아니라 먼 23-0021과 함께 움직인다 (서울시 강남구 센서 18개 중 4개).

실행: python -m src.analysis.t2_case_figure
출력: outputs/figures/fig10_t2_case_0014_0021.png

구성: 왼쪽 지도(센서 위치, 도로, 거리·변화량 상관), 오른쪽 이벤트 5(07-16~17)와 이벤트 8(08-13~14)의 시계열(강우 + 4개 센서).
시계열은 10분 평균 수위를 센서별 훈련 중앙값 기준 상승분으로 바꾼 뒤 이벤트 구간(시작~끝+3시간) 안에서 최소 0, 최대 1로 정규화한 것이다.
크기 비교가 아니라 오르내리는 시점·모양 비교용이며, 첨두 상승(m)은 범례에 표기한다. train+val 구간만 사용(test 미사용).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from src.analysis.spatial_signal import OUT_PAIR, PAIR_CSV  # noqa: E402
from src.data.assign_rain_gauges import node_blocks  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ, TRAIN_END, VAL_END  # noqa: E402
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ  # noqa: E402

TARGETS = {"23-0014": "s2", "23-0021": "s1"}
NEIGHBORS = {"23-0014": "23-0015", "23-0021": "23-0019"}     # 각 센서의 도로거리 최근접 이웃
STYLE = {"23-0014": ("s2", "-"), "23-0015": ("s3", "--"), "23-0021": ("s1", "-"), "23-0019": ("s5", "--")}
EVENTS = [(5, "이벤트 5: 07-16~17 (154.8 mm)"), (8, "이벤트 8: 08-13~14 (115.0 mm)")]
AFTER_H = 3


def normalize_window(x: np.ndarray) -> np.ndarray:
    """창 안 최소를 0, 최대를 1로. 상수 구간이면 0."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def pair_stats() -> dict:
    pw = pd.read_csv(PAIR_CSV)
    pc = pd.read_csv(OUT_PAIR)[["node_a", "node_b", "corr_diff_all"]]
    m = pw.merge(pc, on=["node_a", "node_b"])
    out = {}
    for a, b in [("23-0014", "23-0015"), ("23-0014", "23-0021"), ("23-0021", "23-0019")]:
        r = m[((m.node_a == a) & (m.node_b == b)) | ((m.node_a == b) & (m.node_b == a))].iloc[0]
        out[(a, b)] = dict(road=float(r.road_dist_m), straight=float(r.straight_dist_m), corr=float(r.corr_diff_all))
    return out


def load_series():
    z = np.load(SEWER_NPZ, allow_pickle=True)
    nodes = [str(x) for x in z["node_ids"]]
    blk = node_blocks(z["values"][:VAL_END], z["mask"][:VAL_END])
    bt = pd.DatetimeIndex(z["times"][:VAL_END])[: len(blk) * 10][9::10]
    anom = blk - np.nanmedian(blk[:TRAIN_END // 10], axis=0)
    zr = np.load(RAIN_NPZ, allow_pickle=True)
    gn = [i for i, u in enumerate(zr["gauge_gu"]) if u == "강남구"]
    with np.errstate(all="ignore"):
        mean_gn = np.nanmean(zr["values"][:, gn], axis=1)
    rain = pd.Series(mean_gn, index=pd.DatetimeIndex(zr["times"])).reindex(bt)
    return nodes, anom, bt, rain


def draw_map(ax, C, stats):
    nodes = pd.read_csv(PROCESSED / "gangnam_station_graph_nodes.csv").set_index("node_id")
    rn = pd.read_csv(PROCESSED / "gangnam_road_nodes.csv").set_index("osmid")
    ed = pd.read_csv(PROCESSED / "gangnam_road_edges.csv", usecols=["u", "v"])
    ed = ed[ed.u.isin(rn.index) & ed.v.isin(rn.index)]
    segs = np.stack([np.stack([rn.loc[ed.u, "x"].to_numpy(), rn.loc[ed.u, "y"].to_numpy()], axis=1),
                     np.stack([rn.loc[ed.v, "x"].to_numpy(), rn.loc[ed.v, "y"].to_numpy()], axis=1)], axis=1)
    ax.add_collection(LineCollection(segs, colors="#d6d5d0", linewidths=0.5, zorder=1))
    hi = list(STYLE)
    for n, r in nodes.iterrows():
        if n not in hi:
            ax.scatter(r.lon, r.lat, s=26, color=C["muted"], edgecolor=C["surface"], linewidth=1, zorder=3)
            ax.annotate(n[-4:], (r.lon, r.lat), xytext=(4, 3), textcoords="offset points", fontsize=7, color=C["muted"], zorder=3)
    span_x = nodes.loc[hi, "lon"].max() - nodes.loc[hi, "lon"].min()
    for a, b in stats:
        ax.plot([nodes.loc[a, "lon"], nodes.loc[b, "lon"]], [nodes.loc[a, "lat"], nodes.loc[b, "lat"]], color=C["ink2"], lw=1.1,
                linestyle=(0, (4, 3)), zorder=4)
        mx, my = (nodes.loc[a, "lon"] + nodes.loc[b, "lon"]) / 2, (nodes.loc[a, "lat"] + nodes.loc[b, "lat"]) / 2
        s_ = stats[(a, b)]
        short = s_["straight"] < 300
        text = (f"{s_['road'] / 1000:.2f} km (직선 {s_['straight'] / 1000:.2f})\n변화량 상관 {s_['corr']:.2f}" if s_["road"] > 1000
                else f"{s_['road']:.0f} m\n상관 {s_['corr']:.2f}")
        kw = dict(arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8), xytext=(mx + 0.30 * span_x, my), textcoords="data") if short else {}
        ax.annotate(text, (mx, my), fontsize=9, color=C["ink"], ha="left" if short else "center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc=C["surface"], ec=C["muted"], lw=0.6), **kw)
    label_off = {"23-0014": (-8, 7), "23-0015": (-8, -14), "23-0021": (8, -14), "23-0019": (8, 8)}
    for n, (slot, ls) in STYLE.items():
        r = nodes.loc[n]
        ax.scatter(r.lon, r.lat, s=140, color=C[slot], edgecolor=C["surface"], linewidth=2, zorder=6,
                   marker="o" if n in TARGETS else "s")
        dx, dy = label_off[n]
        ax.annotate(n, (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points", fontsize=10, fontweight="bold", color=C["ink"], zorder=8,
                    ha="right" if dx < 0 else "left", bbox=dict(boxstyle="round,pad=0.15", fc=C["surface"], ec="none", alpha=0.85))
    pts = nodes.loc[hi]
    px, py = (pts.lon.max() - pts.lon.min()) * 0.22, (pts.lat.max() - pts.lat.min()) * 0.22
    ax.set_xlim(pts.lon.min() - px, pts.lon.max() + px)
    ax.set_ylim(pts.lat.min() - py, pts.lat.max() + py)
    ax.set_aspect(1 / np.cos(np.radians(37.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("센서 위치와 도로거리", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(C["muted"])


def main() -> None:
    from src.visualization.make_figures import C, _note, _save
    stats = pair_stats()
    nodes, anom, bt, rain = load_series()
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"]).set_index("event_id")
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.12, 1.7], height_ratios=[0.5, 2.0, 0.45, 0.5, 2.0], wspace=0.08, hspace=0.12)
    draw_map(fig.add_subplot(gs[0:3, 0]), C, stats)
    peaks = {n: [] for n in STYLE}
    for k, (eid, title) in enumerate(EVENTS):
        e = ev.loc[eid]
        m = (bt >= e.start) & (bt <= e.end + pd.Timedelta(hours=AFTER_H))
        t = bt[m]
        r0 = 3 * k
        ax_r = fig.add_subplot(gs[r0, 1])
        ax_s = fig.add_subplot(gs[r0 + 1, 1], sharex=ax_r)
        ax_r.bar(t, rain.to_numpy()[m], width=pd.Timedelta(minutes=9), color=C["muted"], zorder=3)
        ax_r.set_ylabel("강우\n(mm/10분)", fontsize=8)
        ax_r.set_title(title, fontsize=11, loc="left")
        plt.setp(ax_r.get_xticklabels(), visible=False)
        ax_r.grid(axis="x", visible=False)
        for n, (slot, ls) in STYLE.items():
            a_ = anom[m, nodes.index(n)]
            peaks[n].append(float(np.nanmax(a_)))
            ax_s.plot(t, normalize_window(a_), color=C[slot], linestyle=ls, lw=2.0, zorder=3 if n in TARGETS else 2)
        ax_s.set_ylabel("정규화 상승분\n(구간 최소 0, 최대 1)", fontsize=8.5)
        ax_s.set_ylim(-0.05, 1.08)
        ax_s.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d\n%H시"))
    ax_l = fig.add_subplot(gs[3:, 0])
    ax_l.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C[STYLE[n][0]], linestyle=STYLE[n][1], lw=2.4) for n in STYLE]
    role = {"23-0014": "대상(건천, 0 m 90%)", "23-0015": "23-0014의 최근접 이웃", "23-0021": "대상", "23-0019": "23-0021의 최근접 이웃"}
    labels = [f"{n}  {role[n]}   첨두 상승 +{peaks[n][0]:.2f} m / +{peaks[n][1]:.2f} m" for n in STYLE]
    ax_l.legend(handles, labels, loc="upper left", frameon=False, fontsize=9.5, title="센서 (첨두 상승: 이벤트 5 / 이벤트 8)", title_fontsize=9.5,
                alignment="left")
    st = stats
    lines = ["쌍별 거리와 변화량 상관(5분 변화량, train+val)"]
    for (a, b), v in st.items():
        lines.append(f"  {a} ~ {b}: 도로 {v['road']:,.0f} m, 직선 {v['straight']:,.0f} m, 상관 {v['corr']:.2f}")
    ax_l.text(0.0, 0.34, "\n".join(lines), fontsize=9.5, color=C["ink2"], va="top", ha="left", linespacing=1.6)
    fig.suptitle("23-0014의 변동은 가장 가까운 이웃(23-0015, 200 m)보다 더 먼 23-0021(직선 1.4 km)과 더 잘 맞는다 (상관 0.83 대 0.22)",
                 fontsize=13.5, fontweight="bold", y=0.985)
    _note(fig, "서울시 강남구 센서 18개 중 4개(도로거리 최근접 이웃은 점선: 23-0015, 23-0019). 거리는 도로 최단경로, 지도의 점선은 직선 연결. 강우는 강남구 3개소 평균.\n"
               "시계열은 10분 평균 수위를 센서별 훈련 중앙값 기준 상승분으로 바꾼 뒤 이벤트 구간(시작~끝+3시간)에서 0~1로 정규화했다(크기가 아니라 시점·모양 비교).\n"
               "정규화한 모양만 보면 이벤트 8에서는 23-0015도 같은 시점에 첨두가 있고, 차이는 평상시 기저 수위와 변화량 상관에서 크다. 사례 하나이며 인과가 아니다.", y=-0.045)
    p = _save(fig, "fig10_t2_case_0014_0021.png")
    print(p)


if __name__ == "__main__":
    main()
