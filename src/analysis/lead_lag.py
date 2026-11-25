"""센서 간 시차 분석 실행: 사전 고정 규칙(src/analysis/lead_lag_rules.py, docs/20)대로 사건별 '절반까지 오른 시각'을 재고
쌍별 시차·판정 L1·L2와 참고 집계를 만든다.

실행: python -m src.analysis.lead_lag
출력: data/processed/gangnam2026_lead_lag_report.txt, _pairs.csv, _sensors.csv, _events.csv (재생성 가능)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.lead_lag_rules import (BASE_MIN_OBS, COVER_MIN, FREEZE_DATE, K_ROAD, L1_NO, L1_YES, L2_NO, L2_YES, LAG_MIN,
                                         LONG_LAG, MIN_EVENTS, N_BOOT, POST_H, PRE_MIN, RISE_MIN_MM, SEED, consistent_fraction,
                                         first_cross, is_consistent, lag_verdict, pair_summary, rain_t50_block, sign_flip_null,
                                         smooth_rise)
from src.analysis.t2_2026 import OUT_EVENTS, OUT_GAUGE, RAIN_NPZ, TS_NPZ, build_inputs, load_roles
from src.data.clean_nodes import PROCESSED

OUT_REPORT = PROCESSED / "gangnam2026_lead_lag_report.txt"
OUT_PAIRS = PROCESSED / "gangnam2026_lead_lag_pairs.csv"
OUT_SENSORS = PROCESSED / "gangnam2026_lead_lag_sensors.csv"
OUT_EVTAB = PROCESSED / "gangnam2026_lead_lag_events.csv"


def sensor_event_times(nodes: list[str]) -> pd.DataFrame:
    """센서 x 평가 이벤트별 유효 여부, T50·Tpk(창 시작부터 분), 강우 T50, 반응 지연 D, 최대 상승."""
    s = np.load(TS_NPZ)
    ids = [str(x) for x in s["node_ids"]]
    times = pd.DatetimeIndex(s["times"])
    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    gauge = pd.read_csv(OUT_GAUGE, dtype={"gauge_id": str}).set_index("node_id")["gauge_id"]
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    ev = ev[ev.is_eval].reset_index(drop=True)
    rows = []
    for e in ev.itertuples():
        w0, w1 = e.start - pd.Timedelta(minutes=PRE_MIN), e.end + pd.Timedelta(hours=POST_H)
        win = (times >= w0) & (times <= w1)
        pre = (times >= w0) & (times < e.start)
        rwin = (rt > w0) & (rt <= w1 + pd.Timedelta(minutes=9))           # 10분 블록 끝 라벨
        for n in nodes:
            c = ids.index(n)
            lv = np.where(s["mask"][:, c] > 0, s["values"][:, c], np.nan).astype(float)
            x, xp = lv[win], lv[pre]
            cover = float(np.isfinite(x).mean())
            base = float(np.nanmedian(xp)) if np.isfinite(xp).sum() >= BASE_MIN_OBS else np.nan
            row = dict(event_id=e.event_id, node_id=n, gauge_id=gauge[n], cover=cover, base_m=base, peak_mm=np.nan,
                       t50=np.nan, tpk=np.nan, rain_t50=np.nan, valid=False)
            if np.isfinite(base):
                sm = smooth_rise(x, base)
                if np.isfinite(sm).any():
                    row["peak_mm"] = float(np.nanmax(sm))
                    row["t50"] = float(first_cross(sm))
                    row["tpk"] = float(np.nanargmax(sm))
            rb = rain_t50_block(z["values"][rwin, gids.index(gauge[n])].astype(float))
            if rb is not None:
                row["rain_t50"] = float((rt[rwin][rb] - w0) / pd.Timedelta(minutes=1))
            row["valid"] = bool(cover >= COVER_MIN and np.isfinite(base) and row["peak_mm"] >= RISE_MIN_MM
                                and np.isfinite(row["rain_t50"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["delay"] = df.t50 - df.rain_t50
    df["delay_pk"] = df.tpk - df.rain_t50
    return df


def pair_lags(df: pd.DataFrame, nodes: list[str], col: str, base_col: str | None, events=None) -> dict:
    """{(i, j): 사건별 시차 배열} (i<j, 두 센서 모두 유효한 사건). col − base_col 의 차이(강우 보정), base_col None 이면 col 차이."""
    v = df[df.valid]
    rep_id = None
    if events is not None:                                                   # 복원추출: 중복 사건은 다른 사건으로 구분
        parts = [v[v.event_id == e] for e in events]
        rep_id = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
        v = pd.concat(parts)
    val = v[col] - (v[base_col] if base_col else 0.0)
    tab = pd.DataFrame({"event_id": v.event_id.to_numpy() if rep_id is None else rep_id, "node_id": v.node_id.to_numpy(),
                        "x": val.to_numpy()})
    wide = tab.pivot_table(index="event_id", columns="node_id", values="x")
    out = {}
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            na, nb = nodes[a], nodes[b]
            if na not in wide or nb not in wide:
                continue
            d = (wide[nb] - wide[na]).dropna().to_numpy()
            if len(d) >= MIN_EVENTS:
                out[(a, b)] = d
    return out


def road_pairs(dist: np.ndarray) -> set:
    n = len(dist)
    s = set()
    for i in range(n):
        for j in np.argsort(np.where(np.arange(n) == i, np.inf, dist[i]))[:K_ROAD]:
            s.add((min(i, j), max(i, j)))
    return s


def summarize(pairs: dict, road: set) -> dict:
    allp = list(pairs.values())
    rp = [pairs[k] for k in pairs if k in road]
    null_all, null_road = sign_flip_null(allp), sign_flip_null(rp)
    obs_all, obs_road = consistent_fraction(allp), consistent_fraction(rp)
    return dict(n_all=len(allp), n_road=len(rp), obs_all=obs_all, obs_road=obs_road, null_all=null_all, null_road=null_road,
                L1=lag_verdict(obs_road, null_road, L1_YES, L1_NO), L2=lag_verdict(obs_all, null_all, L2_YES, L2_NO))


def main() -> None:
    roles = load_roles()
    nodes = roles["pool"]
    dist = build_inputs(nodes)["dist"]
    road = road_pairs(dist)
    df = sensor_event_times(nodes)
    df.to_csv(OUT_EVTAB, index=False, encoding="utf-8-sig", float_format="%.3f")

    main_pairs = pair_lags(df, nodes, "t50", "rain_t50")
    res = summarize(main_pairs, road)
    raw = summarize(pair_lags(df, nodes, "t50", None), road)
    pk = summarize(pair_lags(df, nodes, "tpk", "rain_t50"), road)

    prow = []
    for (a, b), d in main_pairs.items():
        m, c = pair_summary(d)
        prow.append(dict(node_i=nodes[a], node_j=nodes[b], road_dist_m=dist[a, b], road_nbr=(a, b) in road, n_events=len(d),
                         median_adj_lag_min=m, sign_consistency=c, consistent=is_consistent(m, c),
                         leader=(nodes[a] if m > 0 else nodes[b]) if m != 0 else ""))
    pt = pd.DataFrame(prow)
    pt.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig", float_format="%.3f")

    v = df[df.valid]
    sens = v.groupby("node_id").agg(n_valid=("event_id", "size"), delay_median_min=("delay", "median"),
                                    peak_median_mm=("peak_mm", "median")).reindex(nodes)
    sens["is_target"] = [n in roles["targets"] for n in nodes]
    sens.to_csv(OUT_SENSORS, encoding="utf-8-sig", float_format="%.3f")

    # 참고: |m| 과 도로거리 순위 상관, 이벤트 복원추출
    ev_ids = sorted(df.event_id.unique())
    rho = spearmanr(pt.median_adj_lag_min.abs(), pt.road_dist_m).statistic if len(pt) > 2 else np.nan
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        bp = pair_lags(df, nodes, "t50", "rain_t50", events=list(rng.choice(ev_ids, size=len(ev_ids), replace=True)))
        if len(bp) > 2:
            ks = list(bp)
            boots.append(spearmanr([abs(np.median(bp[k])) for k in ks], [dist[k] for k in ks]).statistic)
    boots = np.array(boots)

    def block(name, r):
        return [f"[{name}]",
                f"  평가 쌍 {r['n_all']}개, 그중 도로 이웃 쌍 {r['n_road']}개",
                f"  일관된 시차 쌍 비율: 도로 이웃 {r['obs_road']:.3f} (우연 95백분위 {np.percentile(r['null_road'], 95):.3f}, "
                f"우연 평균 {r['null_road'].mean():.3f}) / 전체 {r['obs_all']:.3f} (우연 95백분위 {np.percentile(r['null_all'], 95):.3f}, "
                f"우연 평균 {r['null_all'].mean():.3f})"]

    nv = v.groupby("event_id").size().reindex(ev_ids, fill_value=0)
    cons = pt[pt.consistent]
    L = [f"센서 간 시차 분석 (규칙 고정 {FREEZE_DATE}, docs/20). 서울시 강남구 2026 pool {len(nodes)}개, 평가 이벤트 {len(ev_ids)}개",
         f"유효 센서-사건 {int(df.valid.sum())}/{len(df)} (관측률 ≥ {COVER_MIN:.0%}, 최대 상승 ≥ {RISE_MIN_MM:.0f} mm). "
         f"사건별 유효 센서 수: {', '.join(f'{e}:{k}' for e, k in nv.items())}",
         "",
         "[판정]",
         f"  L1 (주) {res['L1']}",
         f"  L2 (보조) {res['L2']}",
         ""] + block("주: 절반까지 오른 시각, 강우 보정", res) + [
         "",
         "[참고 — 판정에 쓰지 않음]",
         f"  일관된 시차 쌍 {len(cons)}개 중 |시차| ≥ {LONG_LAG:.0f}분 {int((cons.median_adj_lag_min.abs() >= LONG_LAG).sum())}개, "
         f"도로 이웃 {int(cons.road_nbr.sum())}개",
         f"  일관된 시차 쌍 기준: |중앙값| ≥ {LAG_MIN:.0f}분, 부호 일관성 ≥ 0.75, 공통 사건 ≥ {MIN_EVENTS}개",
         f"  센서별 반응 지연(강우 절반 시각 → 수위 절반 시각, 사건 중앙값, 분): "
         f"중앙값 {sens.delay_median_min.median():.0f}, 사분위 [{sens.delay_median_min.quantile(.25):.0f}, "
         f"{sens.delay_median_min.quantile(.75):.0f}], 유효 센서 {int(sens.n_valid.notna().sum())}개",
         f"  |시차|와 도로거리 순위 상관 {rho:.3f}, 이벤트 복원추출 95% 구간 [{np.nanpercentile(boots, 2.5):.3f}, "
         f"{np.nanpercentile(boots, 97.5):.3f}] ({len(boots)}회)",
         ""] + block("참고: 강우 보정 전 시차", raw) + [f"  L1 {raw['L1']} / L2 {raw['L2']}", ""] + \
        block("참고: 최고 수위 시각, 강우 보정", pk) + [f"  L1 {pk['L1']} / L2 {pk['L2']}", "",
         "[한계] 이벤트 13개, 쌍끼리 독립 아님, 강우 10분 단위, 어느 쌍도 실제 관로 연결은 아님."]
    if len(cons):
        L += ["", "[일관된 시차 쌍 목록] 앞선 센서 → 뒤따르는 센서, 중앙값(분), 일관성, 공통 사건, 도로거리(m)"]
        for r in cons.sort_values("median_adj_lag_min", key=abs, ascending=False).itertuples():
            follower = r.node_j if r.leader == r.node_i else r.node_i
            L.append(f"  {r.leader} → {follower}  {abs(r.median_adj_lag_min):.0f}분  {r.sign_consistency:.2f}  {r.n_events}  "
                     f"{r.road_dist_m:.0f}{'  (도로 이웃)' if r.road_nbr else ''}")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
