"""연구 진행 결과 그림 4종(PNG, 300 DPI) + HTML 갤러리. 기존 데이터·분석 결과는 읽기만 한다.

실행: python -m src.visualization.make_figures
출력: outputs/figures/*.png, outputs/figures/index.html

1. fig1_spatial_graph_k3.png              도로망 + 관측 지점 18개 + k=3 간선 33개 (도로 최단경로로 그림)
2. fig2_water_levels_full.png / _zoom.png 대표 지점 수위 (결측은 선으로 잇지 않음)
3. fig3_baseline_mae.png / fig3_baseline_rmse.png  기준 모델 2종 test 성능 (GNN 아님)
4. fig4_observation_availability.png      관측 마스크 (전체 3개월 + 장기 결측 확대)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.data.build_road_network import OUT as ROAD_OUT  # noqa: E402
from src.data.build_station_graph import NODES_CSV  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.inspect_water_level import PROJECT_ROOT  # noqa: E402
from src.data.prepare_station_timeseries import OUT_MASK, OUT_VALUES, gap_runs  # noqa: E402
from src.models.baselines import OUT_OVERALL  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "figures"
K3_CSV = PROCESSED / "gangnam_station_graph_k3.csv"
DPI, FIGSIZE = 300, (13.33, 7.5)  # 16:9
# 검증된 기본 팔레트(light) 슬롯 — 고정 순서
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4", "s6": "#008300",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df", "surface": "#fcfcfb",
     "missing": "#e34948", "observed": "#b7d3f6"}
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def _note(fig, text: str, y: float = 0.005):
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=9, color=C["ink2"])


# ---------- 1. 공간 그래프 ----------
def label_offsets(nodes: pd.DataFrame, near_deg: float = 0.004) -> dict[str, tuple[int, int]]:
    """가까운 지점 쌍은 지리 방향대로 라벨을 바깥쪽에 배치: 동서로 떨어지면 서쪽=왼쪽/동쪽=오른쪽,
    남북으로 떨어지면 북쪽=위/남쪽=아래. 라벨이 상대 지점 마커 쪽으로 붙어 헷갈리는 것을 방지."""
    off = {n: (6, 6) for n in nodes.node_id}
    xy = nodes[["lon", "lat"]].to_numpy()
    ids = nodes.node_id.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = xy[j] - xy[i]
            if np.hypot(*d) >= near_deg:
                continue
            if abs(d[0]) >= abs(d[1]):
                west, east = (i, j) if d[0] > 0 else (j, i)
                off[ids[west]], off[ids[east]] = (-6, 6), (6, 6)
            else:
                south, north = (i, j) if d[1] > 0 else (j, i)
                off[ids[north]], off[ids[south]] = (6, 6), (6, -6)
    return off


def fig_spatial_graph() -> list[Path]:
    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(K3_CSV)
    G = ox.load_graphml(ROAD_OUT["network.graphml"])
    U = G.to_undirected()
    rn = dict(zip(nodes.node_id, nodes.road_node))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for u, v, data in U.edges(data=True):
        if "geometry" in data:
            x, y = data["geometry"].xy
        else:
            x, y = [U.nodes[u]["x"], U.nodes[v]["x"]], [U.nodes[u]["y"], U.nodes[v]["y"]]
        ax.plot(x, y, color="#d6d5d0", lw=0.4, zorder=1)
    for r in edges.itertuples():
        path = nx.shortest_path(U, rn[r.node_a], rn[r.node_b], weight="length")
        ax.plot([U.nodes[n]["x"] for n in path], [U.nodes[n]["y"] for n in path],
                color=C["s1"], lw=2.0 if r.mutual else 1.2, alpha=0.85, zorder=2, solid_capstyle="round")
    styles = {"ADDRESS": (C["ink"], "o", "주소 기반 좌표"),
              "FACILITY_PROXY": (C["s2"], "s", "시설 위치 대용 좌표 (23-0007, 23-0014)"),
              "EDGE_FAR": (C["s3"], "^", "도로 구간까지 거리 큼 — 검토 (23-0017)")}
    offsets = label_offsets(nodes)
    for r in nodes.itertuples():
        key = "FACILITY_PROXY" if r.is_facility_proxy else "EDGE_FAR" if r.edge_far else "ADDRESS"
        col, mk, _ = styles[key]
        ax.scatter(r.lon, r.lat, s=70, c=col, marker=mk, edgecolors=C["surface"], linewidths=1.5, zorder=6)
        dx, dy = offsets[r.node_id]
        ax.annotate(r.node_id, (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=C["ink"], zorder=5, ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
                    bbox=dict(boxstyle="round,pad=0.15", fc=C["surface"], ec="none", alpha=0.8))
    pad = 0.004
    ax.set_xlim(nodes.lon.min() - pad, nodes.lon.max() + pad)
    ax.set_ylim(nodes.lat.min() - pad, nodes.lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(nodes.lat.mean())))
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(False)
    ax.set_title("Gangnam Monitoring Station Spatial Graph (k=3)", loc="left")
    handles = [Line2D([], [], color="#d6d5d0", lw=1.5, label="OSM 도로망 (drive)"),
               Line2D([], [], color=C["s1"], lw=2.0, label="k=3 간선 — 상호 이웃 (굵게)"),
               Line2D([], [], color=C["s1"], lw=1.2, label="k=3 간선 — 한쪽 이웃")]
    handles += [Line2D([], [], color="none", marker=mk, markerfacecolor=col, markeredgecolor=C["surface"],
                       markersize=9, label=lab) for col, mk, lab in styles.values()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    _note(fig, f"관측 지점 {len(nodes)}개, 간선 {len(edges)}개 (도로망 최단거리 기준 k-NN, 무방향). "
               "간선은 실제 하수관로 연결이 아니라 도로망 근접 관계(연구용 근사)이며 흐름 방향을 나타내지 않음.\n"
               "좌표 미확정 23-0011(검색 결과 없음), 23-0018(후보 복수)은 그래프에서 제외. 간선은 도로 최단경로를 따라 그림.",
          y=-0.06)
    return [_save(fig, "fig1_spatial_graph_k3.png")]


# ---------- 2. 수위 시계열 ----------
def select_representatives(values: pd.DataFrame, mask: pd.DataFrame) -> dict[str, str]:
    """사전 고정 규칙으로 대표 지점 선정: 최고수위 / 표준편차 중앙값(전형) / 최소 변동 / 0 m 비율 최대 / 최장 결측."""
    std, mx, zero = values.std(), values.max(), (values == 0).mean()
    longest = pd.Series({c: (gap_runs(mask[c].to_numpy()).max() if (mask[c] == 0).any() else 0) for c in mask})
    picks = {}
    for label, node in (("최고 수위", mx.idxmax()),
                        ("전형적 변동(표준편차 중앙값)", (std - std.median()).abs().idxmin()),
                        ("최소 변동", std.idxmin()),
                        ("0 m 비율 최대", zero.idxmax()),
                        ("최장 연속 결측", longest.idxmax())):
        picks.setdefault(node, label)
    return picks


def _shade_missing(ax, mask_col: pd.Series):
    m = mask_col.to_numpy()
    x = np.diff(np.concatenate([[1], m, [1]]))
    idx = mask_col.index
    for s, e in zip(np.where(x == -1)[0], np.where(x == 1)[0]):
        ax.axvspan(idx[s], idx[min(e, len(idx) - 1)], color=C["missing"], alpha=0.18, lw=0, zorder=0)


def _ts_figure(values, mask, picks, start=None, end=None, title_suffix=""):
    v = values.loc[start:end] if start else values
    m = mask.loc[start:end] if start else mask
    fig, axes = plt.subplots(len(picks), 1, figsize=FIGSIZE, sharex=True)
    for ax, (node, why) in zip(axes, picks.items()):
        ax.plot(v.index, v[node], color=C["s1"], lw=0.7)  # NaN에서 선이 끊김(보간 없음)
        _shade_missing(ax, m[node])
        ax.set_ylabel("m", rotation=0, labelpad=10)
        ax.set_ylim(bottom=min(0, v[node].min()))
        ax.text(0.005, 0.92, f"{node}  ·  {why}  ·  max {v[node].max():.2f} m", transform=ax.transAxes,
                fontsize=9, color=C["ink"], va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc=C["surface"], ec="none", alpha=0.85))
        ax.tick_params(labelsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("Date (2025)")
    axes[0].set_title(f"Observed Sewer Water Levels (Jun–Aug 2025){title_suffix}", loc="left")
    axes[0].legend(handles=[Line2D([], [], color=C["s1"], lw=1.2, label="관측 수위 (1분)"),
                            Patch(color=C["missing"], alpha=0.25, label="결측 구간 (선 연결 안 함)")],
                   loc="upper right", frameon=False, fontsize=8)
    return fig, axes


def fig_water_levels() -> list[Path]:
    values = pd.read_csv(OUT_VALUES, index_col=0, parse_dates=True)
    mask = pd.read_csv(OUT_MASK, index_col=0, parse_dates=True)
    picks = select_representatives(values, mask)
    fig, _ = _ts_figure(values, mask, picks)
    _note(fig, "대표 지점은 사전 규칙(최고 수위·표준편차 중앙값·최소 변동·0 m 비율 최대·최장 결측)으로 선정. "
               "패널마다 y축 범위가 다름. 원본 1분 관측값 그대로이며 보간하지 않음.", y=-0.03)
    out = [_save(fig, "fig2_water_levels_full.png")]
    day = values.median(axis=1).resample("D").max().idxmax()  # 18개 지점 중앙값 일 최고수위 날짜
    start, end = day - pd.Timedelta("2D"), day + pd.Timedelta("6D")
    fig, _ = _ts_figure(values, mask, picks, start, end, f" — zoom {start:%m-%d}–{end:%m-%d}")
    _note(fig, f"확대 구간: 18개 지점 중앙값 기준 일 최고수위 날짜({day:%m-%d}) 전 2일~후 6일. "
               "23-0009의 장기 결측(07-19 10:31부터 약 51.5시간)이 포함됨.", y=-0.03)
    out.append(_save(fig, "fig2_water_levels_zoom.png"))
    return out


# ---------- 3. 기준 모델 ----------
SUBSET_LABEL = {"all": "All test pairs", "high_q95": "Target ≥ train P95 (per node)",
                "high_q99": "Target ≥ train P99 (per node)"}
MODEL_LABEL = {"persistence": "Last observation", "node_ridge_ar": "Per-node Ridge AR"}


def fig_baselines() -> list[Path]:
    o = pd.read_csv(OUT_OVERALL)
    t = o[o.split == "test"]
    out = []
    for metric in ("MAE", "RMSE"):
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
        for ax, sub in zip(axes, SUBSET_LABEL):
            d = t[t.subset == sub]
            x = np.arange(2)
            for k, (model, col) in enumerate((("persistence", C["s1"]), ("node_ridge_ar", C["s2"]))):
                vals = [d[(d.horizon_min == h) & (d.model == model)][metric].item() for h in (10, 30)]
                bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=MODEL_LABEL[model],
                              edgecolor=C["surface"], linewidth=2)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                            color=C["ink"])
                # CSV 값과 막대 높이 일치 확인
                assert np.allclose([b.get_height() for b in bars], vals)
            n = d[d.model == "persistence"].set_index("horizon_min")["n_pairs"]
            ax.set_xticks(x, [f"10-min ahead\n(n={n[10]:,})", f"30-min ahead\n(n={n[30]:,})"])
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # 0부터 시작(차이 과장 없음)
            ax.set_title(SUBSET_LABEL[sub], fontsize=10, loc="left")
            ax.set_ylabel(f"{metric} (m)")
            ax.grid(axis="x", visible=False)
        axes[0].legend(frameon=False, loc="upper left", fontsize=9)
        fig.suptitle(f"Baseline Model Comparison - {metric}", x=0.01, ha="left", fontweight="bold", fontsize=13)
        _note(fig, "기준 모델 성능이며 GNN 결과가 아님. Test 구간(2025-08-18 04:48 ~ 08-31), 두 모델 동일 평가쌍. "
                   "단위 m, y축은 0부터 시작. P95/P99는 노드별 학습 구간 관측값 분위수(사전 고정).\n"
                   f"출처: {OUT_OVERALL.relative_to(PROJECT_ROOT).as_posix()}", y=-0.07)
        out.append(_save(fig, f"fig3_baseline_{metric.lower()}.png"))
    return out


# ---------- 4. 관측 마스크 ----------
def bin_any_missing(mask: pd.DataFrame, freq: str) -> pd.DataFrame:
    """구간 안에 1분이라도 결측이면 0(결측), 모두 관측이면 1."""
    return mask.resample(freq).min()


def fig_availability() -> list[Path]:
    mask = pd.read_csv(OUT_MASK, index_col=0, parse_dates=True)
    assert set(np.unique(mask.values)) <= {0, 1}
    hourly = bin_any_missing(mask, "3h")  # 3시간 칸: 행 ≥2px 보장(1시간 칸은 렌더링 다운샘플로 단일 행 결측이 사라짐)
    zs, ze = pd.Timestamp("2025-07-19 00:00"), pd.Timestamp("2025-07-22 00:00")
    zoom = mask.loc[zs:ze - pd.Timedelta("1min")]
    cmap = ListedColormap([C["missing"], C["observed"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})
    for ax, m, title in ((a1, hourly, "Full period (3-hour cells)"),
                         (a2, zoom, "Zoom 07-19 – 07-21 (1-min cells)")):
        t0, t1 = mdates.date2num(m.index[0]), mdates.date2num(m.index[-1])
        ax.imshow(m.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                  extent=[-0.5, m.shape[1] - 0.5, t1, t0])
        ax.yaxis_date()
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d" if m is hourly else "%m-%d %H:%M"))
        ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=7)
        ax.set_xlabel("Monitoring station")
        ax.set_ylabel("Time")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(False)
    total_missing = int((mask == 0).sum().sum())
    fig.legend(handles=[Patch(color=C["observed"], label="Observed (mask = 1)"),
                        Patch(color=C["missing"], label="Missing (mask = 0)")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Observation Availability by Monitoring Station", x=0.01, ha="left", fontweight="bold", fontsize=13)
    _note(fig, f"원본 관측 마스크(1=관측, 0=결측). 전체 {mask.size:,}칸 중 결측 {total_missing:,}칸 "
               f"({total_missing / mask.size:.2%}). 왼쪽은 3시간 칸 — 1분이라도 결측이면 결측으로 표시(짧은 결측이 "
               "과대 표시될 수 있음). 오른쪽은 1분 해상도.", y=-0.08)
    return [_save(fig, "fig4_observation_availability.png")]


def write_gallery(paths: list[Path]) -> Path:
    cards = "\n".join(
        f'<figure><a href="{p.name}"><img src="{p.name}" alt="{p.stem}" loading="lazy"></a>'
        f"<figcaption>{p.name}</figcaption></figure>" for p in paths)
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gangnam Figures</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --line:#383835; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, "Malgun Gothic", sans-serif; }}
h1 {{ font-size:18px; margin:0 0 4px; }} p {{ color:var(--ink2); margin:0 0 16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 520px), 1fr)); gap:16px; }}
figure {{ margin:0; border:1px solid var(--line); border-radius:8px; padding:8px; background:#fcfcfb; }}
img {{ width:100%; height:auto; display:block; }} figcaption {{ color:#52514e; font-size:12px; margin-top:6px; }}
</style></head><body>
<h1>강남구 하수관로 수위 연구 — 결과 그림</h1>
<p>모든 그림은 data/processed의 기존 결과에서 생성. 기준 모델 성능은 GNN 결과가 아님. 그래프 간선은 도로망 근접 관계(실제 하수관로 아님).</p>
<div class="grid">
{cards}
</div></body></html>
"""
    p = OUT_DIR / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


def main() -> None:
    paths = []
    for f in (fig_spatial_graph, fig_water_levels, fig_baselines, fig_availability):
        paths += f()
        print(f"  saved: {', '.join(p.name for p in paths[-2:])}")
    g = write_gallery(paths)
    print("\n".join(str(p.relative_to(PROJECT_ROOT)) for p in paths + [g]))


if __name__ == "__main__":
    main()
