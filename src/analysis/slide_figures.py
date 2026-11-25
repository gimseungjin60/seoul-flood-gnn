"""중간발표 슬라이드용 그림 4장 (글씨를 슬라이드 폭에서 읽히는 크기로 키운 버전). 수치는 기존 결과 파일/모듈의 값만 쓰고 새로 계산하지 않는다.

실행: python -m src.analysis.slide_figures
출력: outputs/figures/slide/{s07_rain_events,s09_model_error,s10_idw_diff,s11_distance_similarity,
      b03_loo_ratio,b04_sensor_gain,b05_case_map,b05_case_series}.png

표시 크기(px)에 맞춰 figsize 를 정하고 글씨는 17 pt 안팎(슬라이드에서 약 24 px 이상)으로 둔다.
서울시 강남구 센서 18개(2025-06-01~08-31), test 구간 미사용.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.rain_events_figure import BIG_MM, START, split_bounds  # noqa: E402
from src.analysis.spatial_signal import OUT_PAIR  # noqa: E402
from src.analysis.t2_loso import MODELS, OUT_PRED  # noqa: E402
from src.analysis.t2_loso_bootstrap import OUT_CI, event_error_sums  # noqa: E402
from src.analysis.t2_loso_sensitivity import run_scenario, subset  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.inspect_water_level import PROJECT_ROOT  # noqa: E402
from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ  # noqa: E402
from src.visualization.make_figures import C  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "figures" / "slide"
FS = 17
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "AppleGothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": FS, "axes.titlesize": FS + 1,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return p


def rain_events():
    z = np.load(RAIN_NPZ, allow_pickle=True)
    t = pd.DatetimeIndex(z["times"])
    gn = [i for i, u in enumerate(z["gauge_gu"]) if u == "강남구"]
    with np.errstate(all="ignore"):
        rain10 = pd.Series(np.nanmean(z["values"][:, gn], axis=1), index=t)
    rain10 = rain10[(rain10.index >= START) & (rain10.index < pd.Timestamp("2025-09-01"))]
    daily = rain10.resample("D").sum(min_count=1)
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    fig, ax = plt.subplots(figsize=(15.6, 4.8))
    tint = {"train": "#E9EFF9", "val": "#FBEFE6", "test": "#E7F2EC"}
    label = {"train": "train 70%", "val": "val 15%", "test": "test 15%"}
    ymax = float(np.nanmax(daily.to_numpy()))
    top = ymax * 1.45
    for name, (a, b) in split_bounds().items():
        ax.axvspan(a, b, color=tint[name], zorder=0)
        ax.text(a + pd.Timedelta(days=1), top * 0.97, label[name], ha="left", va="top", fontsize=FS + 1, color=C["ink2"], fontweight="bold")
    ax.bar(daily.index + pd.Timedelta(hours=12), daily.to_numpy(), width=pd.Timedelta(hours=20), color=C["muted"], zorder=3, linewidth=0)
    for r in ev.itertuples():
        seg = daily[(daily.index >= r.start.normalize()) & (daily.index <= r.end.normalize())]
        big = r.total_mm >= BIG_MM
        ax.annotate(f"{int(r.event_id)}\n{r.total_mm:.0f}", (seg.idxmax() + pd.Timedelta(hours=12), float(seg.max())), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=FS if big else FS - 2,
                    color=C["s2"] if big else C["ink"], fontweight="bold" if big else "normal", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc=C["surface"], ec=C["s2"] if big else "none", lw=1.5 if big else 0, alpha=0.92))
    ax.set_ylim(0, top)
    ax.set_ylabel("일 강우량 (mm)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
    ax.set_xlim(START, pd.Timestamp("2025-09-01"))
    ax.grid(axis="x", visible=False)
    return _save(fig, "s07_rain_events.png")


def model_error():
    ci = pd.read_csv(OUT_CI).set_index("model")
    order = ["mean_others", "idw_p1", "ridge_rain_time", "ridge_rain_time_nbr", "hgb_rain_time_nbr"]
    names = ["평균", "거리가중\n(IDW)", "강우 모델", "강우+이웃\n(선형)", "강우+이웃\n(트리)"]
    v = np.array([ci.loc[m, "macro_event_mae_m"] for m in order]) * 1000
    lo = v - np.array([ci.loc[m, "ci_lo"] for m in order]) * 1000
    hi = np.array([ci.loc[m, "ci_hi"] for m in order]) * 1000 - v
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    colors = [C["s1"], C["s1"], C["muted"], C["s1"], C["s1"]]
    x = np.arange(len(order))
    ax.bar(x, v, width=0.62, color=colors, zorder=3)
    ax.errorbar(x, v, yerr=[lo, hi], fmt="none", ecolor=C["ink"], elinewidth=2, capsize=6, zorder=4)
    for xi, vi, hi_ in zip(x, v, hi):
        ax.text(xi, vi + hi_ + 0.5, f"{vi:.1f}", ha="center", va="bottom", fontsize=FS + 3, fontweight="bold", color=C["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("호우 구간 평균 오차 (mm)")
    ax.set_ylim(0, (v + hi).max() * 1.2)
    ax.grid(axis="x", visible=False)
    return _save(fig, "s09_model_error.png")


def idw_diff():
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    pred = {m: z[f"pred_{m}"].astype(float) for m in MODELS}
    nodes = [str(x) for x in z["nodes"]]
    S, N, ev_ids = event_error_sums(pred, z["actual_m"].astype(float), bt, pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"]))
    scen = {"전체 이벤트 8개": run_scenario(*subset(S, N, ev_ids, nodes)[:2]),
            "큰 사건 2개 제외\n(이벤트 6개)": run_scenario(*subset(S, N, ev_ids, nodes, drop_events=(5, 8))[:2])}
    rows = []
    for sc, r in scen.items():
        rows.append((sc, "강우+이웃(선형)", r["d4_point"] * 1000, r["d4"][0] * 1000, r["d4"][1] * 1000, C["s1"], "o"))
        rows.append((sc, "강우+이웃(트리)", r["d5_point"] * 1000, r["d5"][0] * 1000, r["d5"][1] * 1000, C["s2"], "D"))
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ypos = [3.4, 2.5, 1.0, 0.1]
    for y, (sc, name, p, lo, hi, col, mk) in zip(ypos, rows):
        ax.plot([lo, hi], [y, y], color=col, lw=4, solid_capstyle="round", zorder=3)
        ax.scatter([p], [y], s=170, color=col, marker=mk, edgecolor=C["surface"], linewidth=2, zorder=4)
        ax.text(hi + 0.08, y, f"{p:+.1f}", va="center", fontsize=FS, fontweight="bold", color=C["ink"])
    ax.axvline(0, color=C["ink"], lw=1.6, linestyle="--", zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r[0]}\n{r[1]}" if i % 2 == 0 else f"\n{r[1]}" for i, r in enumerate(rows)])
    ax.set_ylim(-0.6, 4.1)
    ax.set_xlim(-1.9, 3.6)
    ax.set_xlabel("IDW와의 평균 오차 차이 (mm)  ·  선은 오차 범위(95%)")
    ax.text(-0.08, -0.52, "← 학습 모델이 작음", fontsize=FS - 2, color=C["ink2"], va="bottom", ha="right")
    ax.text(0.08, -0.52, "IDW가 작음 →", fontsize=FS - 2, color=C["ink2"], va="bottom", ha="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, "s10_idw_diff.png")


def distance_similarity():
    """강우 제거 후 센서 쌍 153개의 거리와 변화 유사도. 계산은 spatial_signal_rain_adjusted 와 같은 경로를 따르고 Spearman 이 기존 값과 같은지 확인한다."""
    from src.analysis.spatial_signal import PAIR_CSV, corr_matrix, spearman
    from src.analysis.spatial_signal_rain_adjusted import remove_rain
    from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV, node_blocks
    from src.data.prepare_rainfall import TRAIN_END, VAL_END
    from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ
    s = np.load(SEWER_NPZ, allow_pickle=True)
    nodes = [str(n) for n in s["node_ids"]]
    n = len(nodes)
    blk = node_blocks(s["values"][:VAL_END], s["mask"][:VAL_END])
    bt = pd.DatetimeIndex(s["times"][:VAL_END])[: len(blk) * 10][9::10]
    z = np.load(RAIN_NPZ, allow_pickle=True)
    gids = [str(g) for g in z["gauge_ids"]]
    rain_df = pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["times"]), columns=gids).reindex(bt)
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    dist = np.zeros((n, n))
    nid = {k: i for i, k in enumerate(nodes)}
    for r in pd.read_csv(PAIR_CSV).itertuples():
        dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = r.road_dist_m
    dy = np.vstack([np.full((1, n), np.nan), np.diff(blk, axis=0)])
    fit_rows = np.arange(len(blk)) < TRAIN_END // 10
    res = np.full_like(dy, np.nan)
    for j, nd in enumerate(nodes):
        res[:, j], _ = remove_rain(dy[:, j], rain_df[asg.loc[nd, "gauge_id"]].to_numpy(), fit_rows)
    c = corr_matrix(res)
    iu = np.triu_indices(n, 1)
    rho = spearman(dist[iu], np.nan_to_num(c[iu], nan=0.0))
    assert abs(rho - (-0.226)) < 0.002, f"강우 제거 후 Spearman 이 기존 값(-0.226)과 다르다: {rho:.3f}"
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.scatter(dist[iu] / 1000, c[iu], s=70, color=C["s1"], edgecolor=C["surface"], linewidth=1.5, zorder=3)
    ax.axhline(0, color=C["muted"], lw=1.2)
    ax.set_xlabel("도로 최단거리 (km)")
    ax.set_ylabel("센서 쌍의 수위 변화 유사도")
    return _save(fig, "s11_distance_similarity.png")


def backup_loo():
    """백업 3: 센서 하나를 뺀 추정의 평균 MAE / 상수 추정 MAE (1차 진단 결과 CSV 값 그대로)."""
    from src.analysis.spatial_signal import OUT_ALL
    t = pd.read_csv(OUT_ALL).set_index("model")
    order = ["mean_others", "idw_p1", "idw_p2", "knn3", "ridge_all"]
    names = {"mean_others": "이웃 평균", "idw_p1": "IDW (p=1)", "idw_p2": "IDW (p=2)", "knn3": "도로 최근접 3개", "ridge_all": "Ridge(이력 사용)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True)
    for ax, col, ttl in zip(axes, ["const_ratio_all", "const_ratio_high"], ["전체", "고수위"]):
        vals = [t.loc[m, col] for m in order]
        ax.barh([names[m] for m in order], vals, color=C["s1"], height=0.58, zorder=3)
        ax.axvline(1.0, color=C["ink2"], lw=1.6, zorder=4)
        ax.axvline(0.9, color=C["s2"], lw=1.6, linestyle="--", zorder=4)
        ax.set_title(ttl, fontweight="bold")
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("상수 추정 대비 오차 비율")
        ax.grid(axis="y", visible=False)
        for k, v in enumerate(vals):
            ax.text(v + 0.015, k, f"{v:.2f}", va="center", fontsize=FS, color=C["ink"], fontweight="bold")
    axes[0].invert_yaxis()
    return _save(fig, "b03_loo_ratio.png")


def backup_sensor_gain():
    """백업 4: 센서별 호우 구간 오차 개선율(강우 모델 대비). 센서별 CSV 값 그대로."""
    from src.analysis.t2_loso import OUT_SENSOR, IMPROVE
    d = pd.read_csv(OUT_SENSOR)
    d = d.assign(best=d[["gain_ridge_rain_time_nbr_vs_3", "gain_hgb_rain_time_nbr_vs_3"]].max(axis=1)).sort_values("best")
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    y = np.arange(len(d))
    h = 0.4
    ax.barh(y - h / 2, d.gain_ridge_rain_time_nbr_vs_3 * 100, height=h, color=C["s1"], label="강우+이웃(선형)", zorder=3)
    ax.barh(y + h / 2, d.gain_hgb_rain_time_nbr_vs_3 * 100, height=h, color=C["s2"], label="강우+이웃(트리)", zorder=3)
    ax.axvline(0, color=C["ink2"], lw=1.4, zorder=4)
    ax.axvline(IMPROVE * 100, color=C["ink"], lw=1.4, linestyle="--", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(d.sensor, fontsize=FS - 4)
    ax.set_xlabel("오차 개선율 (%), 강우 모델 대비")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=FS - 3)
    return _save(fig, "b04_sensor_gain.png")


CASE = {"23-0014": (C["s2"], "-"), "23-0015": (C["s3"], "--"), "23-0021": (C["s1"], "-"), "23-0019": (C["s5"], "--")}


def case_map():
    """백업 5 왼쪽: 센서 위치와 거리, 변화 유사도(5분 변화량 상관). 값은 기존 쌍별 CSV 그대로."""
    from matplotlib.collections import LineCollection
    from src.analysis.t2_case_figure import pair_stats
    stats = pair_stats()
    nodes = pd.read_csv(PROCESSED / "gangnam_station_graph_nodes.csv").set_index("node_id")
    rn = pd.read_csv(PROCESSED / "gangnam_road_nodes.csv").set_index("osmid")
    ed = pd.read_csv(PROCESSED / "gangnam_road_edges.csv", usecols=["u", "v"])
    ed = ed[ed.u.isin(rn.index) & ed.v.isin(rn.index)]
    segs = np.stack([np.stack([rn.loc[ed.u, "x"].to_numpy(), rn.loc[ed.u, "y"].to_numpy()], axis=1),
                     np.stack([rn.loc[ed.v, "x"].to_numpy(), rn.loc[ed.v, "y"].to_numpy()], axis=1)], axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.add_collection(LineCollection(segs, colors="#d6d5d0", linewidths=0.6, zorder=1))
    hi = list(CASE)
    span_x = nodes.loc[hi, "lon"].max() - nodes.loc[hi, "lon"].min()
    for a, b in stats:
        ax.plot([nodes.loc[a, "lon"], nodes.loc[b, "lon"]], [nodes.loc[a, "lat"], nodes.loc[b, "lat"]], color=C["ink2"], lw=1.4,
                linestyle=(0, (4, 3)), zorder=4)
        mx, my = (nodes.loc[a, "lon"] + nodes.loc[b, "lon"]) / 2, (nodes.loc[a, "lat"] + nodes.loc[b, "lat"]) / 2
        v = stats[(a, b)]
        short = v["straight"] < 300
        txt = f"{v['road'] / 1000:.2f} km · {v['corr']:.2f}" if v["road"] > 1000 else f"{v['road']:.0f} m · {v['corr']:.2f}"
        kw = dict(arrowprops=dict(arrowstyle="-", color=C["muted"], lw=1.0), xytext=(mx + 0.38 * span_x, my), textcoords="data") if short else {}
        ax.annotate(txt, (mx, my), fontsize=FS - 4, color=C["ink"], ha="left" if short else "center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc=C["surface"], ec=C["muted"], lw=0.8), **kw)
    off = {"23-0014": (-7, 7), "23-0015": (-7, -16), "23-0021": (8, -16), "23-0019": (8, 8)}
    for n, (col, ls) in CASE.items():
        r = nodes.loc[n]
        ax.scatter(r.lon, r.lat, s=190, color=col, edgecolor=C["surface"], linewidth=2, zorder=6, marker="o" if n in ("23-0014", "23-0021") else "s")
        dx, dy = off[n]
        ax.annotate(n, (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points", fontsize=FS - 4, fontweight="bold", color=C["ink"], zorder=8,
                    ha="right" if dx < 0 else "left", bbox=dict(boxstyle="round,pad=0.15", fc=C["surface"], ec="none", alpha=0.85))
    pts = nodes.loc[hi]
    px, py = (pts.lon.max() - pts.lon.min()) * 0.25, (pts.lat.max() - pts.lat.min()) * 0.25
    ax.set_xlim(pts.lon.min() - px, pts.lon.max() + px)
    ax.set_ylim(pts.lat.min() - py, pts.lat.max() + py)
    ax.set_aspect(1 / np.cos(np.radians(37.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(C["muted"])
    return _save(fig, "b05_case_map.png")


def case_series():
    """백업 5 오른쪽: 이벤트 5, 8 구간의 정규화 수위 시계열(구간 안 최소 0, 최대 1)."""
    from src.analysis.t2_case_figure import load_series, normalize_window
    nodes, anom, bt, _ = load_series()
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"]).set_index("event_id")
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 5.9))
    for ax, eid, ttl in zip(axes, (5, 8), ("이벤트 5 (07-16~17)", "이벤트 8 (08-13~14)")):
        e = ev.loc[eid]
        m = (bt >= e.start) & (bt <= e.end + pd.Timedelta(hours=3))
        for n, (col, ls) in CASE.items():
            ax.plot(bt[m], normalize_window(anom[m, nodes.index(n)]), color=col, linestyle=ls, lw=2.2, label=n, zorder=3 if n in ("23-0014", "23-0021") else 2)
        ax.set_title(ttl, fontsize=FS, loc="left", fontweight="bold")
        ax.set_ylabel("정규화 상승분")
        ax.set_ylim(-0.05, 1.32)
        ax.set_yticks([0, 0.5, 1.0])
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6 if eid == 5 else 4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H시"))
        ax.tick_params(axis="x", labelsize=FS - 4)
    axes[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=FS - 4)
    fig.tight_layout()
    return _save(fig, "b05_case_series.png")


def main() -> None:
    for f in (rain_events, model_error, idw_diff, distance_similarity, backup_loo, backup_sensor_gain, case_map, case_series):
        print(f())


if __name__ == "__main__":
    main()
