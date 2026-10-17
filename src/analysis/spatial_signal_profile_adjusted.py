"""생활하수 일주기(시간대 x 요일 프로파일)를 먼저 제거한 뒤 강우 제거 잔차 상관·강우 회귀 설명력을 다시 계산 (GNN 아님).

실행: python -m src.analysis.spatial_signal_profile_adjusted
출력: data/processed/gangnam_spatial_signal_profile_adjusted_report.txt

배경: 일주기(생활하수)가 센서들에 공통으로 작용하면 공통 상관이 생긴다. 이를 빼면 강우·거리 구조가 달라지는지 확인한다.

사전 고정 규칙 (결과를 보기 전에 정함. 결과를 본 뒤 바꾸지 않는다)
  데이터: train+val 구간만(test 미사용). 해상도 두 가지: 10분 평균(관측 5개 이상), 1시간 평균(10분 블록 6개 중 4개 이상).
  프로파일: 센서별 '요일 x 시' 168개 빈의 평균(train 구간 관측 블록만). 수준에서 프로파일을 빼고 그 차분(dA)을 쓴다.
            민감도로 평균 대신 중앙값 프로파일도 계산한다(train 호우가 평균을 끌어올릴 수 있어서).
  강우 제거: docs/07 과 같은 방식(센서별 최근접 강우량계 log1p 3슬롯 선형, train 적합). 1시간 해상도는 시간 합계 mm 를 쓴다.
  지표: Mantel 순열검정(1000회) Spearman(거리 vs 변화량 상관, 사전 기준 <= -0.30 & p<0.05), 상관 중앙값, 강우 회귀 R2 중앙값,
        3-NN 일치율(도로 3-NN vs 상관 상위 3)과 노드 라벨 순열검정(1000회) p. 센서 쌍은 독립이 아니므로 순열은 노드 라벨 단위.
  비교: 같은 해상도에서 (a) 원 변화량 (b) 강우 제거 (c) 프로파일 제거 (d) 프로파일+강우 제거 를 나란히 본다.
  판정: 프로파일 제거(d)에서 Spearman <= -0.30 이고 p < 0.05 이면 '일주기가 거리 구조를 가리고 있었다', 아니면 '결론 불변(사전 기준 미달)'.
한계: 프로파일 평균이 train 호우에 오염될 수 있다(중앙값 민감도로 점검). 강우 제거는 선형 3슬롯이다.
"""
import time

import numpy as np
import pandas as pd

from src.analysis.spatial_signal import PAIR_CSV, corr_matrix, mantel
from src.analysis.spatial_signal_rain_adjusted import remove_rain
from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV, node_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_NPZ as RAIN_NPZ, TRAIN_END, VAL_END
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ

REPORT = PROCESSED / "gangnam_spatial_signal_profile_adjusted_report.txt"
N_PERM = 1000
SEED = 0


def hourly(blk10: np.ndarray, min_valid: int = 4) -> np.ndarray:
    """(B, N) 10분 블록 → (B//6, N) 1시간 평균(유효 블록 min_valid개 이상). 첫 블록이 정시에 시작해야 한다."""
    b = blk10.shape[0] // 6 * 6
    x = blk10[:b].reshape(-1, 6, blk10.shape[1])
    ok = np.isfinite(x)
    cnt = ok.sum(axis=1)
    m = np.where(ok, x, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    return np.where(cnt >= min_valid, m, np.nan)


def hourly_rain(rain10: np.ndarray, min_valid: int = 4) -> np.ndarray:
    """(B,) 10분 mm → (B//6,) 시간 합계 mm. 유효 슬롯이 min_valid개 미만이면 NaN, 일부 결측은 유효 평균 x 6 으로 보정."""
    b = len(rain10) // 6 * 6
    x = rain10[:b].reshape(-1, 6)
    ok = np.isfinite(x)
    cnt = ok.sum(axis=1)
    s = np.where(ok, x, 0.0).sum(axis=1) * 6 / np.maximum(cnt, 1)
    return np.where(cnt >= min_valid, s, np.nan)


def hour_of_week(start_times: pd.DatetimeIndex) -> np.ndarray:
    return (start_times.dayofweek * 24 + start_times.hour).to_numpy()


def week_profile(level: np.ndarray, how: np.ndarray, train_rows: np.ndarray, stat: str = "mean") -> np.ndarray:
    """(B, N) 수준 → (168, N) 요일x시 프로파일(train 관측 블록만). 빈이 비면 그 센서의 train 전체 통계로 채운다."""
    n = level.shape[1]
    prof = np.full((168, n), np.nan)
    fn = np.nanmean if stat == "mean" else np.nanmedian
    for b in range(168):
        rows = train_rows & (how == b)
        if rows.any():
            with np.errstate(all="ignore"):
                prof[b] = fn(np.where(np.isfinite(level[rows]), level[rows], np.nan), axis=0)
    with np.errstate(all="ignore"):
        fallback = fn(np.where(train_rows[:, None], level, np.nan), axis=0)
    return np.where(np.isfinite(prof), prof, fallback[None, :])


def diff(x: np.ndarray) -> np.ndarray:
    return np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)])


def overlap_stat(dist: np.ndarray, corr: np.ndarray, k: int = 3) -> float:
    """노드마다 (도로거리 최근접 k) 와 (상관 상위 k) 의 겹침 비율 평균."""
    n = dist.shape[0]
    o = []
    for i in range(n):
        road = set(np.argsort(np.where(np.arange(n) == i, np.inf, dist[i]))[:k])
        cc = np.where(np.arange(n) == i, -np.inf, np.nan_to_num(corr[i], nan=-9))
        o.append(len(road & set(np.argsort(-cc)[:k])) / k)
    return float(np.mean(o))


def overlap_perm_test(dist: np.ndarray, corr: np.ndarray, n_perm: int = N_PERM, seed: int = SEED, k: int = 3):
    """거리행렬의 노드 라벨을 무작위로 섞어(센서 쌍의 비독립성 유지) 관측 일치율이 우연보다 큰지. 반환: (관측, 순열 평균, p)."""
    obs = overlap_stat(dist, corr, k)
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    sims = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        sims.append(overlap_stat(dist[p][:, p], corr, k))
    sims = np.array(sims)
    return obs, float(sims.mean()), float((np.sum(sims >= obs) + 1) / (n_perm + 1))


def evaluate(name, mat, dist, event, r2=None):
    iu = np.triu_indices(dist.shape[0], 1)
    c_all, c_ev, c_calm = corr_matrix(mat), corr_matrix(mat, event), corr_matrix(mat, ~event)
    rho, p = mantel(dist, np.nan_to_num(c_all, nan=0.0), n_perm=N_PERM)
    if np.isfinite(c_ev[iu]).sum() < 0.8 * len(iu[0]):      # 이벤트 블록이 적어 상관을 대부분 계산할 수 없으면(최소 표본 200) 검정하지 않는다
        rho_e = p_e = np.nan
    else:
        rho_e, p_e = mantel(dist, np.nan_to_num(c_ev, nan=0.0), n_perm=N_PERM)
    ov, ov_rand, ov_p = overlap_perm_test(dist, c_all)
    return {"name": name, "rho": rho, "p": p, "rho_ev": rho_e, "p_ev": p_e, "med": float(np.nanmedian(c_all[iu])),
            "med_calm": float(np.nanmedian(c_calm[iu])),
            "med_ev": float(np.nanmedian(c_ev[iu])) if np.isfinite(c_ev[iu]).any() else np.nan,
            "r2": float(np.nanmedian(r2)) if r2 is not None else np.nan, "overlap": ov, "overlap_rand": ov_rand,
            "overlap_p": ov_p, "ratio": ov / (3 / 17)}


def run_resolution(label, level, rain, how, tr_end, dist, nodes, asg_gauges):
    """level (B,N), rain (B,G dict gauge→(B,) 해당 해상도 mm), how (B,) 시간대x요일 빈."""
    n = level.shape[1]
    train_rows = np.arange(len(level)) < tr_end
    anom = level - np.nanmedian(level[:tr_end], axis=0)
    with np.errstate(all="ignore"):
        net = np.nanmedian(anom, axis=1)
    event = np.isfinite(net) & (net >= np.nanquantile(net, 0.95))
    series = {}
    for stat in ("mean", "median"):
        prof = week_profile(level, how, train_rows, stat)
        series[stat] = diff(level - prof[how])
    dy = diff(level)

    def rain_remove(mat):
        res = np.full_like(mat, np.nan)
        r2 = []
        for j in range(n):
            res[:, j], v = remove_rain(mat[:, j], rain[asg_gauges[j]], train_rows)
            r2.append(v)
        return res, np.array(r2)
    dy_r, r2_a = rain_remove(dy)
    dA_r, r2_c = rain_remove(series["mean"])
    dM_r, r2_m = rain_remove(series["median"])
    rows = [evaluate("(a) 원 변화량", dy, dist, event),
            evaluate("(b) 강우 제거", dy_r, dist, event, r2_a),
            evaluate("(c) 프로파일(평균) 제거", series["mean"], dist, event),
            evaluate("(d) 프로파일(평균)+강우 제거", dA_r, dist, event, r2_c),
            evaluate("(e) 프로파일(중앙값)+강우 제거 [민감도]", dM_r, dist, event, r2_m)]
    return rows, int(event.sum()), len(level), r2_a, r2_c


def main() -> None:
    t0 = time.time()
    s = np.load(SEWER_NPZ, allow_pickle=True)
    nodes = [str(x) for x in s["node_ids"]]
    n = len(nodes)
    blk10 = node_blocks(s["values"][:VAL_END], s["mask"][:VAL_END])
    t10 = pd.DatetimeIndex(s["times"][:VAL_END])[: len(blk10) * 10]
    bt10 = t10[9::10]                                                       # 10분 블록 끝 라벨
    z = np.load(RAIN_NPZ, allow_pickle=True)
    gids = [str(g) for g in z["gauge_ids"]]
    rain_df = pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["times"]), columns=gids).reindex(bt10)
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    gauge_of = [asg.loc[nd, "gauge_id"] for nd in nodes]
    nid = {k: i for i, k in enumerate(nodes)}
    dist = np.zeros((n, n))
    for r in pd.read_csv(PAIR_CSV).itertuples():
        dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = r.road_dist_m

    out = {}
    # 10분
    how10 = hour_of_week(bt10 - pd.Timedelta(minutes=9))
    rain10 = {g: rain_df[g].to_numpy() for g in set(gauge_of)}
    out["10분"] = run_resolution("10분", blk10, rain10, how10, TRAIN_END // 10, dist, nodes, gauge_of)
    # 1시간
    h = hourly(blk10)
    bt1 = bt10[: len(h) * 6][5::6]
    how1 = hour_of_week(bt1 - pd.Timedelta(minutes=59))
    rain1 = {g: hourly_rain(rain10[g])[: len(h)] for g in rain10}
    out["1시간"] = run_resolution("1시간", h, rain1, how1, TRAIN_END // 60, dist, nodes, gauge_of)

    L = []
    w = L.append
    w("[강남] 일주기(시간대 x 요일) 프로파일 제거 후 공간 진단 (GNN 학습 없음, test 구간 미사용)")
    w("=" * 78)
    w("※ 사전 고정 규칙은 src/analysis/spatial_signal_profile_adjusted.py 상단 docstring 참조(결과 확인 전 작성).")
    w(f"Mantel·일치율 순열검정 모두 {N_PERM}회, 노드 라벨 순열(센서 쌍의 비독립성 반영). 3-NN 무작위 기대 일치율 {3 / 17:.1%}.")
    for res, (rows, n_ev, n_blk, r2_a, r2_c) in out.items():
        w("")
        w(f"[{res} 해상도] 블록 {n_blk:,}개, 이벤트 블록 {n_ev}개")
        w(f"  {'변형':34s} {'Spearman':>9s} {'p':>7s} {'(이벤트)':>9s} {'p':>7s} {'상관중앙(전체/평상/이벤트)':>27s} {'R2중앙':>7s} {'3-NN일치':>9s} {'배수':>5s} {'순열p':>7s}")
        for r in rows:
            r2s = f"{r['r2']:.3f}" if np.isfinite(r["r2"]) else "  -  "
            ev_s = f"{r['rho_ev']:+9.3f} {r['p_ev']:7.4f}" if np.isfinite(r["rho_ev"]) else f"{'n/a':>9s} {'n/a':>7s}"
            med_ev = f"{r['med_ev']:.3f}" if np.isfinite(r["med_ev"]) else "n/a"
            w(f"  {r['name']:34s} {r['rho']:+9.3f} {r['p']:7.4f} {ev_s} "
              f"  {r['med']:.3f}/{r['med_calm']:.3f}/{med_ev}   {r2s:>7s} {r['overlap']:9.1%} {r['ratio']:5.2f} {r['overlap_p']:7.4f}")
        w(f"  강우 회귀 R2 중앙값: 프로파일 제거 전 {np.nanmedian(r2_a):.3f} → 후 {np.nanmedian(r2_c):.3f} "
          f"(노드별 전 {np.round(r2_a, 2).tolist()})")
        w(f"                               노드별 후 {np.round(r2_c, 2).tolist()}")
        d = rows[3]
        w(f"  사전 규칙 판정(d 기준): Spearman <= -0.30 & p<0.05 → {bool(d['rho'] <= -0.30 and d['p'] < 0.05)}")
    w("")
    w("주의")
    w("  - 프로파일 평균은 train 호우에 오염될 수 있다. 중앙값 프로파일(e)로 민감도를 확인했다.")
    w("  - 일주기 제거 후 R2가 낮아지면 이전의 강우 설명력 일부가 일주기와 겹쳤던 것이다. 높아지면 일주기가 강우 신호를 가리고 있었다.")
    w("  - 1시간 해상도는 블록 수가 6배 적어 상관 추정이 더 불안정하다. 이벤트 블록은 94개뿐이라 상관 계산 최소 표본(200)에 못 미쳐 이벤트 열은 n/a다(검정하지 않음).")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
