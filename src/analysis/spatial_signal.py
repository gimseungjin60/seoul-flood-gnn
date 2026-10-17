"""공간 신호 진단 (GNN 학습 전): 센서 간 수위 변동이 도로거리와 관련 있는가, 이웃이 정보를 주는가.

실행: python -m src.analysis.spatial_signal
출력: data/processed/gangnam_spatial_signal_{report.txt,pair_corr.csv,loo_by_node.csv,loo_overall.csv}
      outputs/figures/fig5_distance_correlation.png, fig6_loo_vs_constant.png

사전 고정 규칙 (결과를 보기 전에 정함. 결과를 본 뒤 바꾸지 않는다)
  데이터: train+val 구간만 사용 (idx < 112,608). test 구간은 읽지 않는다. 5분 평균(관측 3개 이상일 때만 유효).
  이상값 anomaly = 5분 평균 - 그 노드의 train 중앙값.
  이벤트 블록 = 전 노드 anomaly의 중앙값이 (train+val 전체 블록 분포의) 상위 5% 이상인 블록.
  C1 거리 감쇠: 5분 변화량(diff) 상관과 도로거리의 Spearman <= -0.30 이고 Mantel 순열검정(노드 라벨 5000회) p < 0.05
  C2 이웃 도움: leave-one-node-out(같은 시각 나머지 17개 노드로 추정, val 구간)에서 IDW(p=1,2)/도로 3-NN 중 최선의
     노드 평균 MAE가 상수(노드 train 중앙값) 대비 10% 이상 낮고, 고수위 부분(대상 노드 train q95 이상)에서도 10% 이상 낮음
  C3 거리가 의미 있음: IDW(p=1)의 MAE가 거리를 무작위로 섞은 IDW 100회의 5% 하위 분위수보다 낮음
  L  선형(Ridge, 17개 노드 전체 입력)이 상수 대비 10% 이상 낮으면 '선형 공간 정보 있음'
  판정: C1·C2·C3 모두 참 -> 도로 근접 그래프 근거 있음 / C2 참, C3 거짓 -> 공간 정보는 있으나 거리 기반 근접 그래프는 근거 없음
        C2 거짓, L 참 -> 거리 무관한 선형 공통 신호 / 모두 거짓 -> 센서 간 공간 이득 없음
한계: 대상 노드의 과거 중앙값(train)을 안다고 가정한다(센서가 과거에 있었던 지점). 센서가 전혀 없던 지점은 이 가정이 깨진다.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.prepare_station_timeseries import OUT_NPZ  # noqa: E402

B = 5                      # 평균 블록(분)
MIN_OBS = 3                # 블록 유효 최소 관측 수
TRAIN_END, VAL_END = 92_736, 112_608   # 분 단위 인덱스 (70/15/15 분할과 동일)
LAGS = list(range(-12, 13))            # +-60분 (5분 단위)
SEED = 0
N_MANTEL, N_PERM_IDW = 5000, 100
THRESH = 0.10
PAIR_CSV = PROCESSED / "gangnam_station_pairwise_road_distance.csv"
OUT_REPORT = PROCESSED / "gangnam_spatial_signal_report.txt"
OUT_PAIR = PROCESSED / "gangnam_spatial_signal_pair_corr.csv"
OUT_NODE = PROCESSED / "gangnam_spatial_signal_loo_by_node.csv"
OUT_ALL = PROCESSED / "gangnam_spatial_signal_loo_overall.csv"


def block_mean(values: np.ndarray, b: int = B, min_obs: int = MIN_OBS) -> np.ndarray:
    """(T, N) 1분 값(NaN 결측) -> (T//b, N) b분 평균. 관측이 min_obs개 미만이면 NaN."""
    t = values.shape[0] // b * b
    v = values[:t].reshape(-1, b, values.shape[1])
    ok = np.isfinite(v)
    cnt = ok.sum(axis=1)
    m = np.where(ok, v, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    m[cnt < min_obs] = np.nan
    return m


def pair_corr(a: np.ndarray, b: np.ndarray, min_n: int = 200) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def corr_matrix(x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    n = x.shape[1]
    xs = x if rows is None else x[rows]
    c = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            c[i, j] = c[j, i] = 1.0 if i == j else pair_corr(xs[:, i], xs[:, j])
    return c


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def mantel(dist: np.ndarray, corr: np.ndarray, n_perm: int = N_MANTEL, seed: int = SEED) -> tuple[float, float]:
    """거리행렬 vs 상관행렬 Spearman + 노드 라벨 순열 p값(한쪽: 음의 상관이 관측값 이하일 확률)."""
    n = dist.shape[0]
    iu = np.triu_indices(n, 1)
    obs = spearman(dist[iu], corr[iu])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        cnt += spearman(dist[p][:, p][iu], corr[iu]) <= obs
    return obs, (cnt + 1) / (n_perm + 1)


def best_lags(d: np.ndarray, lags=LAGS) -> pd.DataFrame:
    """노드 쌍별 최대 교차상관 시차(+: j가 i보다 늦음)와 lag0 대비 상관."""
    n = d.shape[1]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            cs = []
            for k in lags:
                a, b = (d[:len(d) - k, i], d[k:, j]) if k >= 0 else (d[-k:, i], d[:len(d) + k, j])
                cs.append(pair_corr(a, b))
            cs = np.array(cs)
            if np.isnan(cs).all():
                continue
            kbest = int(np.nanargmax(cs))
            rows.append({"i": i, "j": j, "best_lag_min": lags[kbest] * B, "corr_best": cs[kbest],
                         "corr_lag0": cs[lags.index(0)]})
    return pd.DataFrame(rows)


def wmean(anom: np.ndarray, w: np.ndarray) -> np.ndarray:
    """anom (T, M) NaN 허용, w (M,) -> (T,) 가중평균. 유효 이웃이 없으면 NaN."""
    ok = np.isfinite(anom)
    s = ok @ w
    num = np.where(ok, anom, 0.0) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, num / s, np.nan)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float):
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def _mae(pred, yt, m):
    return float(np.abs(pred[m] - yt[m]).mean()) if m.any() else np.nan


def loo_eval(x: np.ndarray, dist: np.ndarray, tr_end: int, val_end: int, nodes: list[str],
             lam: float = 100.0, n_perm: int = N_PERM_IDW, seed: int = SEED) -> pd.DataFrame:
    """x: (Tb, N) 블록 평균. 노드별·모델별·부분집합별 MAE/RMSE (perm_idw_p1_k = 거리 섞은 IDW k번째)."""
    n = x.shape[1]
    med = np.nanmedian(x[:tr_end], axis=0)
    anom = x - med
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        oth = [j for j in range(n) if j != i]
        d_i = dist[i, oth]
        a_o = anom[:, oth]
        y = x[:, i]
        q95 = np.nanquantile(y[:tr_end], 0.95)
        est = {"const": np.zeros(len(x)),                      # anomaly 0 = 노드 train 중앙값
               "mean_others": wmean(a_o, np.ones(len(oth))),
               "idw_p1": wmean(a_o, 1.0 / d_i),
               "idw_p2": wmean(a_o, 1.0 / d_i ** 2),
               "knn3": wmean(a_o, (np.argsort(np.argsort(d_i)) < 3).astype(float))}
        ok_tr = np.isfinite(y[:tr_end])
        a_fill = np.where(np.isfinite(a_o), a_o, 0.0)
        f = ridge_fit(a_fill[:tr_end][ok_tr], anom[:tr_end, i][ok_tr], lam)
        est["ridge_all"] = f(a_fill)
        va = np.arange(tr_end, val_end)
        va = va[np.isfinite(y[va]) & np.isfinite(a_o[va]).any(axis=1)]
        high = y[va] >= q95
        yt = y[va]
        for name, e in est.items():
            pred = med[i] + e[va]
            ok = np.isfinite(pred)
            for subset, m in (("all", ok), ("high_q95", ok & high)):
                rows.append({"node": nodes[i], "model": name, "subset": subset, "n": int(m.sum()),
                             "MAE": _mae(pred, yt, m),
                             "RMSE": float(np.sqrt(((pred[m] - yt[m]) ** 2).mean())) if m.any() else np.nan})
        for k in range(n_perm):
            pred = med[i] + wmean(a_o, 1.0 / rng.permutation(d_i))[va]
            ok = np.isfinite(pred)
            for subset, m in (("all", ok), ("high_q95", ok & high)):
                rows.append({"node": nodes[i], "model": f"perm_idw_p1_{k}", "subset": subset, "n": 0,
                             "MAE": _mae(pred, yt, m), "RMSE": np.nan})
    return pd.DataFrame(rows)


def summarize(loo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    is_perm = loo.model.str.startswith("perm_idw")
    macro = loo[~is_perm].groupby(["model", "subset"])["MAE"].mean().unstack()
    macro["const_ratio_all"] = macro["all"] / macro.loc["const", "all"]
    macro["const_ratio_high"] = macro["high_q95"] / macro.loc["const", "high_q95"]
    perm = loo[is_perm].assign(k=lambda d: d.model.str.rsplit("_", n=1).str[-1].astype(int))
    pm = perm.groupby(["k", "subset"])["MAE"].mean().unstack()    # k번째 순열의 노드 평균
    stats = {s: (float(pm[s].mean()), float(pm[s].quantile(0.05)), float(pm[s].quantile(0.95))) for s in pm.columns}
    return macro, stats


def verdict(rho: float, p: float, macro: pd.DataFrame, perm_stats: dict, thr: float = THRESH) -> dict:
    best = min(["idw_p1", "idw_p2", "knn3"], key=lambda m: macro.loc[m, "all"])
    c1 = bool(rho <= -0.30 and p < 0.05)
    c2 = bool(macro.loc[best, "const_ratio_all"] <= 1 - thr and macro.loc[best, "const_ratio_high"] <= 1 - thr)
    c3 = bool(macro.loc["idw_p1", "all"] < perm_stats["all"][1])
    lin = bool(macro.loc["ridge_all", "const_ratio_all"] <= 1 - thr)
    if c1 and c2 and c3:
        v = "도로 근접 그래프의 근거 있음"
    elif c2 and not c3:
        v = "공간 정보는 있으나 거리 기반 근접 그래프의 근거는 없음 (상관 기반 그래프 등 대안 검토)"
    elif not c2 and lin:
        v = "거리와 무관한 선형 공통 신호만 있음 (근접 그래프 근거 없음)"
    elif not c2:
        v = "센서 간 공간 이득 없음 (이 조건에서는)"
    else:
        v = "혼합 결과: C1/C3 불일치, 항목별로 해석"
    return {"best": best, "C1": c1, "C2": c2, "C3": c3, "linear": lin, "verdict": v}


def build_figures(dist, c_calm, c_event, macro, nodes):
    from src.visualization.make_figures import C, _note, _save
    iu = np.triu_indices(len(nodes), 1)
    d = dist[iu] / 1000
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(d, c_calm[iu], s=36, color=C["s1"], edgecolor=C["surface"], linewidth=1.5, label="평상시 블록", zorder=3)
    ax.scatter(d, c_event[iu], s=36, color=C["s2"], edgecolor=C["surface"], linewidth=1.5,
               label="이벤트 블록(상위 5%)", zorder=3)
    ax.axhline(0, color=C["muted"], linewidth=1)
    ax.set_xlabel("도로 최단거리 (km)")
    ax.set_ylabel("5분 수위 변화량 상관계수")
    ax.set_title("센서 간 변동 상관은 도로거리와 함께 줄어드는가")
    ax.legend(frameon=False, loc="upper right")
    _note(fig, "서울시 강남구 센서 18개. 5분 평균의 변화량 상관, train+val 구간, 153개 지점 쌍.", y=-0.03)
    _save(fig, "fig5_distance_correlation.png")

    order = ["mean_others", "idw_p1", "idw_p2", "knn3", "ridge_all"]
    labels = {"mean_others": "17개 평균", "idw_p1": "IDW p=1", "idw_p2": "IDW p=2", "knn3": "도로 3-NN",
              "ridge_all": "Ridge(17개)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, col, ttl in zip(axes, ["const_ratio_all", "const_ratio_high"], ["전체", "고수위(대상 노드 q95 이상)"]):
        vals = [macro.loc[m, col] for m in order]
        ax.barh([labels[m] for m in order], vals, color=C["s1"], height=0.55, zorder=3)
        ax.axvline(1.0, color=C["ink2"], linewidth=1.2, zorder=4)
        ax.axvline(1 - THRESH, color=C["s2"], linewidth=1.2, linestyle="--", zorder=4)
        ax.set_title(ttl)
        ax.set_xlabel("MAE / 상수 추정 MAE (1 미만이면 상수보다 좋음)")
        ax.grid(axis="y", visible=False)
        for k, v in enumerate(vals):
            ax.text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=9, color=C["ink2"])
    axes[0].invert_yaxis()
    _note(fig, "서울시 강남구 센서 18개. val 구간 leave-one-node-out, 노드 평균. 점선=사전 기준(10% 개선), 실선=상수 추정(노드 train 중앙값).", y=-0.03)
    _save(fig, "fig6_loo_vs_constant.png")


def main() -> None:
    t0 = time.time()
    z = np.load(OUT_NPZ, allow_pickle=True)
    nodes = [str(s) for s in z["node_ids"]]
    values = z["values"][:VAL_END].astype(np.float64)   # test 구간은 읽지 않는다
    values[z["mask"][:VAL_END] == 0] = np.nan
    tr_end, val_end = TRAIN_END // B, VAL_END // B
    x = block_mean(values)[:val_end]
    nid = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    dist = np.zeros((n, n))
    for r in pd.read_csv(PAIR_CSV).itertuples():
        dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = r.road_dist_m

    anom = x - np.nanmedian(x[:tr_end], axis=0)
    net = np.nanmedian(anom, axis=1)
    thr_ev = np.nanquantile(net, 0.95)
    event = np.isfinite(net) & (net >= thr_ev)
    d = np.diff(x, axis=0)
    ev_d = event[1:] & event[:-1]
    c_all, c_calm, c_ev, c_lvl = corr_matrix(d), corr_matrix(d, ~ev_d), corr_matrix(d, ev_d), corr_matrix(anom)
    rho, p = mantel(dist, np.nan_to_num(c_all, nan=0.0))
    rho_ev, p_ev = mantel(dist, np.nan_to_num(c_ev, nan=0.0))
    rho_lvl, p_lvl = mantel(dist, np.nan_to_num(c_lvl, nan=0.0))
    lg = best_lags(d)
    loo = loo_eval(x, dist, tr_end, val_end, nodes)
    macro, perm_stats = summarize(loo)
    v = verdict(rho, p, macro, perm_stats)

    iu = np.triu_indices(n, 1)
    pd.DataFrame({"node_a": [nodes[i] for i in iu[0]], "node_b": [nodes[j] for j in iu[1]],
                  "road_dist_m": dist[iu], "corr_diff_all": c_all[iu], "corr_diff_calm": c_calm[iu],
                  "corr_diff_event": c_ev[iu], "corr_level_anom": c_lvl[iu]}).to_csv(
        OUT_PAIR, index=False, encoding="utf-8-sig", float_format="%.4f")
    loo[~loo.model.str.startswith("perm_idw")].to_csv(OUT_NODE, index=False, encoding="utf-8-sig", float_format="%.5f")
    macro.to_csv(OUT_ALL, encoding="utf-8-sig", float_format="%.5f")
    build_figures(dist, c_calm, c_ev, macro, nodes)

    ov = []
    for i in range(n):
        road = set(np.argsort(np.where(np.arange(n) == i, np.inf, dist[i]))[:3])
        cc = np.where(np.arange(n) == i, -np.inf, np.nan_to_num(c_all[i], nan=-9))
        ov.append(len(road & set(np.argsort(-cc)[:3])) / 3)
    gain = lg.corr_best - lg.corr_lag0

    L = []
    w = L.append
    w("[강남] 공간 신호 진단 리포트 (GNN 학습 없음, test 구간 미사용)")
    w("=" * 70)
    w("※ 사전 고정 규칙은 src/analysis/spatial_signal.py 상단 docstring 참조(결과 확인 전 작성).")
    w(f"데이터: {n}노드, 5분 평균 블록 {val_end:,}개 (train {tr_end:,} / val {val_end - tr_end:,}), 이벤트 블록 {int(event.sum())}개 "
      f"(전 노드 anomaly 중앙값 >= {thr_ev:.4f} m)")
    w("")
    w("1. 도로거리 vs 변동 상관 (153쌍, Mantel 순열검정 5000회)")
    w(f"  5분 변화량 상관, 전체:   Spearman {rho:+.3f}, p={p:.4f}")
    w(f"  5분 변화량 상관, 이벤트: Spearman {rho_ev:+.3f}, p={p_ev:.4f}")
    w(f"  수준(anomaly) 상관:      Spearman {rho_lvl:+.3f}, p={p_lvl:.4f}")
    w(f"  상관계수 분포(변화량, 전체): 중앙값 {np.nanmedian(c_all[iu]):.3f}, 최소 {np.nanmin(c_all[iu]):.3f}, 최대 {np.nanmax(c_all[iu]):.3f}")
    near = np.argsort(dist[iu])[:5]
    w("  가장 가까운 5쌍 상관(변화량): " + ", ".join(f"{nodes[iu[0][k]]}-{nodes[iu[1][k]]} {c_all[iu][k]:.2f}" for k in near))
    w("")
    w("2. 시차 상관 (+-60분, 5분 단위, 변화량)")
    w(f"  최대 상관 시차가 0이 아닌 쌍: {(lg.best_lag_min != 0).mean():.1%} ({len(lg)}쌍), "
      f"lag0 대비 상관 증가 중앙값 {gain.median():+.3f}, 최대 {gain.max():+.3f}")
    w("  (5분 해상도라 5분 미만 지연은 보이지 않음. 증가가 크면 상·하류 방향성의 단서)")
    w("")
    w("3. 이웃 일치: 도로 3-NN vs 상관 상위 3 (변화량)")
    w(f"  평균 겹침 {np.mean(ov):.1%} (무작위 기대값 {3 / 17:.1%})")
    w("")
    w("4. Leave-one-node-out 추정 (val 구간, 같은 시각 나머지 17개 노드로 추정, 노드 평균 MAE, m)")
    w(macro.round(4).to_string())
    w(f"  거리 섞은 IDW(p=1) 100회: 전체 MAE 평균 {perm_stats['all'][0]:.4f} "
      f"(5~95% {perm_stats['all'][1]:.4f}~{perm_stats['all'][2]:.4f}), 고수위 평균 {perm_stats['high_q95'][0]:.4f}")
    w("")
    w("5. 사전 규칙 판정")
    w(f"  C1 거리 감쇠(Spearman<=-0.30 & p<0.05): {v['C1']}")
    w(f"  C2 이웃 도움(최선 {v['best']}, 전체·고수위 모두 10% 이상 개선): {v['C2']}")
    w(f"  C3 거리 의미(IDW p=1 < 거리 섞은 5% 분위): {v['C3']}")
    w(f"  L  선형(Ridge 17개) 10% 이상 개선: {v['linear']}")
    w(f"  판정: {v['verdict']}")
    w("")
    w("주의")
    w("  - 대상 노드의 train 중앙값을 안다고 가정한 추정이다. 센서가 전혀 없던 지점의 추정은 이보다 어렵다.")
    w("  - 평가는 val 구간(약 14일)이며 고수위 표본은 소수 사건에 몰려 있다. 5분 블록은 서로 독립이 아니다.")
    w("  - 23-0014(0 m 90%)처럼 변동이 거의 없는 노드는 상관이 불안정하다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
