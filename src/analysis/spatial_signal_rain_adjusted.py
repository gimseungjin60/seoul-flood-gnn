"""강우 영향을 제거한 뒤에도 센서 간 변동 상관이 도로거리와 관련 있는가 (공간 신호 진단 2차, GNN 아님).

실행: python -m src.analysis.spatial_signal_rain_adjusted
출력: data/processed/gangnam_spatial_signal_rain_adjusted_report.txt, outputs/figures/fig7_rain_adjusted_correlation.png

배경: 1차 진단(docs/05)에서 변동 상관은 도로거리와 약하게만 관련됐다(Spearman -0.17). 같은 비를 동시에 맞아서 생긴 상관이
      거리 구조를 가렸을 가능성을 확인한다.

사전 고정 규칙 (결과를 보기 전에 정함. 결과를 본 뒤 바꾸지 않는다)
  데이터: train+val 구간만(test 미사용), 10분 평균 블록(관측 5개 이상), 변화량 dy = 블록 평균의 10분 차분.
  강우 제거: 노드별로 dy 를 [log1p(mm) 같은 슬롯, 1슬롯 전, 2슬롯 전] (센서별 최근접 강우량계) 와 절편에 OLS 로 적합(train 블록만),
            잔차 = dy - 적합값 (train+val 전체에 적용). 강우 결측 블록은 잔차 NaN. 선형 제거라 비선형 강우 영향은 남는다.
  C1r: 잔차 변화량 상관과 도로거리의 Spearman <= -0.30 이고 Mantel 순열검정(5000회) p < 0.05
  변화: 강우 제거 전 대비 Spearman 이 0.10 이상 더 음수이면 '강우가 거리 구조를 가리고 있었다'
  판정: C1r 참 -> 강우 제거 후 거리 감쇠 확인(근접 그래프 근거 보강)
        C1r 거짓 & 상관 중앙값이 크게 줄면 -> 상관 대부분이 공통 강우 반응, 거리 구조는 따로 없음
        C1r 거짓 & 상관 중앙값이 그대로면 -> 강우와 무관한 상관이 있으나 거리로 설명되지 않음
한계: 강우 효과를 선형(3개 슬롯)으로만 제거했다. 센서별 최근접 강우량계 하나만 쓴다.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.spatial_signal import PAIR_CSV, corr_matrix, mantel  # noqa: E402
from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV, node_blocks  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.prepare_rainfall import OUT_NPZ as RAIN_NPZ, TRAIN_END, VAL_END  # noqa: E402
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ  # noqa: E402

REPORT = PROCESSED / "gangnam_spatial_signal_rain_adjusted_report.txt"
LAG_SLOTS = (0, 1, 2)
EVENT_Q = 0.95


def rain_design(rain: np.ndarray) -> np.ndarray:
    """(B,) mm → (B, 3) log1p(mm) 같은 슬롯·1슬롯 전·2슬롯 전. 앞쪽 채울 수 없는 칸은 NaN."""
    r = np.log1p(rain)
    out = np.full((len(r), len(LAG_SLOTS)), np.nan)
    for k, lag in enumerate(LAG_SLOTS):
        out[lag:, k] = r[:len(r) - lag]
    return out


def remove_rain(dy: np.ndarray, rain: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, float]:
    """dy (B,) 를 rain_design 으로 OLS(fit_rows 에서만) 적합해 잔차 반환. 설명된 분산 비율(train)도 반환."""
    X = np.hstack([rain_design(rain), np.ones((len(dy), 1))])
    ok = np.isfinite(dy) & np.isfinite(X).all(axis=1)
    tr = ok & fit_rows
    beta, *_ = np.linalg.lstsq(X[tr], dy[tr], rcond=None)
    res = np.full(len(dy), np.nan)
    res[ok] = dy[ok] - X[ok] @ beta
    r2 = 1 - np.var(dy[tr] - X[tr] @ beta) / np.var(dy[tr]) if np.var(dy[tr]) > 0 else np.nan
    return res, float(r2)


def main() -> None:
    t0 = time.time()
    s = np.load(SEWER_NPZ, allow_pickle=True)
    nodes = [str(n) for n in s["node_ids"]]
    n = len(nodes)
    blk = node_blocks(s["values"][:VAL_END], s["mask"][:VAL_END])         # (B, N), 끝 라벨 :09
    bt = pd.DatetimeIndex(s["times"][:VAL_END])[: len(blk) * 10][9::10]
    tr_end_blk = TRAIN_END // 10
    z = np.load(RAIN_NPZ, allow_pickle=True)
    gids = [str(g) for g in z["gauge_ids"]]
    rain_df = pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["times"]), columns=gids).reindex(bt)
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    dist = np.zeros((n, n))
    nid = {k: i for i, k in enumerate(nodes)}
    for r in pd.read_csv(PAIR_CSV).itertuples():
        dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = r.road_dist_m

    dy = np.vstack([np.full((1, n), np.nan), np.diff(blk, axis=0)])
    fit_rows = np.arange(len(blk)) < tr_end_blk
    res = np.full_like(dy, np.nan)
    r2 = []
    for j, nd in enumerate(nodes):
        rain = rain_df[asg.loc[nd, "gauge_id"]].to_numpy()
        res[:, j], v = remove_rain(dy[:, j], rain, fit_rows)
        r2.append(v)
    # 이벤트 블록 (1차 진단과 같은 방식: 전 노드 anomaly 중앙값 상위 5%)
    anom = blk - np.nanmedian(blk[:tr_end_blk], axis=0)
    with np.errstate(all="ignore"):
        net = np.nanmedian(anom, axis=1)
    event = np.isfinite(net) & (net >= np.nanquantile(net, EVENT_Q))
    ev = event[:]

    iu = np.triu_indices(n, 1)
    out = {}
    for name, mat in (("강우 제거 전(변화량)", dy), ("강우 제거 후(잔차)", res)):
        c_all, c_ev, c_calm = corr_matrix(mat), corr_matrix(mat, ev), corr_matrix(mat, ~ev)
        rho, p = mantel(dist, np.nan_to_num(c_all, nan=0.0))
        rho_e, p_e = mantel(dist, np.nan_to_num(c_ev, nan=0.0))
        out[name] = dict(c_all=c_all, c_ev=c_ev, c_calm=c_calm, rho=rho, p=p, rho_e=rho_e, p_e=p_e,
                         med=float(np.nanmedian(c_all[iu])), med_ev=float(np.nanmedian(c_ev[iu])),
                         med_calm=float(np.nanmedian(c_calm[iu])))
    b0, b1 = out["강우 제거 전(변화량)"], out["강우 제거 후(잔차)"]
    c1r = bool(b1["rho"] <= -0.30 and b1["p"] < 0.05)
    shift = b1["rho"] - b0["rho"]
    masked = bool(shift <= -0.10)

    # 이웃 일치 (도로 3-NN vs 상관 상위 3)
    def overlap(c):
        o = []
        for i in range(n):
            road = set(np.argsort(np.where(np.arange(n) == i, np.inf, dist[i]))[:3])
            cc = np.where(np.arange(n) == i, -np.inf, np.nan_to_num(c[i], nan=-9))
            o.append(len(road & set(np.argsort(-cc)[:3])) / 3)
        return float(np.mean(o))
    ov0, ov1 = overlap(b0["c_all"]), overlap(b1["c_all"])

    if c1r:
        v = "강우 제거 후 거리 감쇠 확인 (근접 그래프 근거 보강)"
    elif b1["med"] < 0.5 * b0["med"]:
        v = "상관 대부분이 공통 강우 반응 (강우 제거 후 거리 구조 없음)"
    else:
        v = "강우와 무관한 상관이 남아 있으나 거리로 설명되지 않음"

    from src.visualization.make_figures import C, _note, _save
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    d = dist[iu] / 1000
    for ax, key, ttl in zip(axes, ("강우 제거 전(변화량)", "강우 제거 후(잔차)"), ("강우 제거 전", "강우 제거 후(잔차)")):
        o = out[key]
        ax.scatter(d, o["c_all"][iu], s=30, color=C["s1"], edgecolor=C["surface"], linewidth=1.2, zorder=3, label="전체 블록")
        ax.scatter(d, o["c_ev"][iu], s=30, color=C["s2"], edgecolor=C["surface"], linewidth=1.2, zorder=3, label="이벤트 블록")
        ax.axhline(0, color=C["muted"], linewidth=1)
        ax.set_title(f"{ttl}  (Spearman {o['rho']:+.2f}, p={o['p']:.3f})")
        ax.set_xlabel("도로 최단거리 (km)")
    axes[0].set_ylabel("10분 수위 변화량 상관계수")
    axes[0].legend(frameon=False, loc="upper right")
    _note(fig, "센서별 최근접 강우(3슬롯, 선형)를 train 구간에서 적합해 제거. train+val, 153개 지점 쌍, 이벤트=전 노드 anomaly 상위 5%.", y=-0.03)
    _save(fig, "fig7_rain_adjusted_correlation.png")

    L = []
    w = L.append
    w("[강남] 공간 신호 진단 2차: 강우 제거 후 (GNN 학습 없음, test 구간 미사용)")
    w("=" * 70)
    w("※ 사전 고정 규칙은 src/analysis/spatial_signal_rain_adjusted.py 상단 docstring 참조(결과 확인 전 작성).")
    w(f"데이터: {n}노드, 10분 블록 {len(blk):,}개 (train {tr_end_blk:,}), 이벤트 블록 {int(ev.sum())}개")
    w(f"강우 제거: 노드별 train OLS 가 설명하는 dy 분산(R2) 중앙값 {np.nanmedian(r2):.3f}, min {np.nanmin(r2):.3f}, max {np.nanmax(r2):.3f}")
    w("  노드별 R2: " + ", ".join(f"{nd} {v:.2f}" for nd, v in zip(nodes, r2)))
    w("")
    w("1. 도로거리 vs 변화량 상관 (153쌍, Mantel 순열검정 5000회)")
    w(f"  {'':22s} {'Spearman(전체)':>15s} {'p':>7s} {'Spearman(이벤트)':>17s} {'p':>7s} {'상관 중앙값(전체/평상시/이벤트)':>30s}")
    for k, o in out.items():
        w(f"  {k:22s} {o['rho']:+15.3f} {o['p']:7.4f} {o['rho_e']:+17.3f} {o['p_e']:7.4f}   {o['med']:.3f} / {o['med_calm']:.3f} / {o['med_ev']:.3f}")
    w(f"  Spearman 변화(전체): {shift:+.3f}")
    w("")
    w(f"2. 이웃 일치 (도로 3-NN vs 상관 상위 3): 강우 제거 전 {ov0:.1%} → 후 {ov1:.1%} (무작위 기대 {3 / 17:.1%})")
    w("")
    w("3. 사전 규칙 판정")
    w(f"  C1r 강우 제거 후 거리 감쇠(Spearman<=-0.30 & p<0.05): {c1r}")
    w(f"  강우가 거리 구조를 가리고 있었는가(Spearman이 0.10 이상 더 음수): {masked}")
    w(f"  판정: {v}")
    w("")
    w("주의")
    w("  - 강우 제거는 선형·3슬롯이다. 비선형 영향이 남으면 '강우와 무관한 상관'으로 잘못 읽힐 수 있다.")
    w("  - 센서별 최근접 강우량계 하나만 쓴다. 강우량계 자체의 관측 오차와 센서-강우량계 거리(193~2,382 m)가 잔차에 남는다.")
    w("  - 23-0014처럼 변화가 거의 없는 노드는 상관이 불안정하다. 평가는 train+val 이며 이벤트 블록은 소수 사건에 몰려 있다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
