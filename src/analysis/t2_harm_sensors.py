"""T2 LOSO에서 이웃 정보가 해가 된 센서(23-0013, 0014, 0015, 0020)의 위치·이웃 거리·결측률·반응 특성 정리 (GNN 아님, test 미사용).

실행: python -m src.analysis.t2_harm_sensors   (먼저 t2_loso 실행)
출력: data/processed/gangnam_t2_harm_sensors.csv, gangnam_t2_harm_sensors_report.txt

여기 나오는 값은 모두 데이터에서 계산한 '사실'이다. 원인은 이 스크립트가 판정하지 않는다(docs/08 에서 추정으로 따로 표시).
비교 맥락으로 나머지 14개 센서의 중앙값을 함께 낸다.
"""
import numpy as np
import pandas as pd

from src.analysis.spatial_signal import OUT_PAIR, PAIR_CSV
from src.analysis.t2_loso import OUT_PRED, OUT_SENSOR, event_blocks
from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV, best_lag_corr, node_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ, VAL_END
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ

HARM = ["23-0013", "23-0014", "23-0015", "23-0020"]
OUT_CSV = PROCESSED / "gangnam_t2_harm_sensors.csv"
REPORT = PROCESSED / "gangnam_t2_harm_sensors_report.txt"


def build_table() -> pd.DataFrame:
    nodes = pd.read_csv(PROCESSED / "gangnam_station_graph_nodes.csv").set_index("node_id")
    geo = pd.read_csv(PROCESSED / "gangnam_geocoding_result.csv").set_index("node_id")
    norm = pd.read_csv(PROCESSED / "gangnam_forecast_norm_stats.csv").set_index("node_id")
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id")
    loso = pd.read_csv(OUT_SENSOR).set_index("sensor")
    pair = pd.read_csv(OUT_PAIR).merge(pd.read_csv(PAIR_CSV)[["node_a", "node_b", "straight_dist_m"]], on=["node_a", "node_b"], how="left")
    ids = nodes.index.tolist()

    z = np.load(SEWER_NPZ, allow_pickle=True)
    s_ids = [str(x) for x in z["node_ids"]]
    mask = z["mask"].astype(bool)
    n_all = mask.shape[0]
    miss_all = {n: 1 - mask[:, j].mean() for j, n in enumerate(s_ids)}
    miss_tv = {n: 1 - mask[:VAL_END, j].mean() for j, n in enumerate(s_ids)}

    # 이웃 거리·상관
    nn = {}
    for n in ids:
        p = pair[(pair.node_a == n) | (pair.node_b == n)].copy()
        p["other"] = np.where(p.node_a == n, p.node_b, p.node_a)
        p = p.sort_values("road_dist_m")
        first = p.iloc[0]
        top = p.sort_values("corr_diff_all", ascending=False).iloc[0]
        nn[n] = dict(nn_id=first.other, nn_road_m=first.road_dist_m, nn_straight_m=first.straight_dist_m,
                     nn_corr=first.corr_diff_all, nn3_mean_road_m=p.road_dist_m.iloc[:3].mean(),
                     max_corr=top.corr_diff_all, max_corr_with=top.other, max_corr_road_m=top.road_dist_m,
                     median_corr=p.corr_diff_all.median())

    # 이벤트 반응 크기와 이벤트 구간 결측
    zp = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(zp["bt"])
    actual = zp["actual_m"].astype(float)
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    ev = event_blocks(bt, events)
    peaks = {n: [] for n in ids}
    for _, idx in ev:
        for j, n in enumerate(ids):
            a = actual[idx, j]
            if np.isfinite(a).any():
                peaks[n].append(np.nanmax(a))

    # 강우와의 관계 (센서별 최근접 강우량계, 10분 변화량 상관)
    blk = node_blocks(z["values"][:VAL_END], z["mask"][:VAL_END])
    bt10 = pd.DatetimeIndex(z["times"][:VAL_END])[: len(blk) * 10][9::10]
    zr = np.load(RAIN_NPZ, allow_pickle=True)
    gids = [str(g) for g in zr["gauge_ids"]]
    rain_df = pd.DataFrame(zr["values"], index=pd.DatetimeIndex(zr["times"]), columns=gids).reindex(bt10)
    dy = np.vstack([np.full((1, blk.shape[1]), np.nan), np.diff(blk, axis=0)])
    rows = []
    for j, n in enumerate(s_ids):
        c, lag = best_lag_corr(rain_df[asg.loc[n, "gauge_id"]].to_numpy(), dy[:, j])
        rows.append({"sensor": n, "address": geo.loc[n, "matched_address"], "place": geo.loc[n, "matched_place"],
                     "district": f"강남구(코드 {int(geo.loc[n, 'district_code'])})", "lat": nodes.loc[n, "lat"], "lon": nodes.loc[n, "lon"],
                     "location_basis": nodes.loc[n, "location_basis"], "facility_proxy": bool(nodes.loc[n, "is_facility_proxy"]),
                     "needs_review": bool(nodes.loc[n, "needs_review"]), "dist_to_road_node_m": nodes.loc[n, "dist_to_road_node_m"],
                     "gauge": asg.loc[n, "gauge_name"], "gauge_gu": asg.loc[n, "gauge_gu"], "gauge_dist_m": asg.loc[n, "dist_m"],
                     "rain_corr_best": c, "rain_corr_lag_min": lag,
                     "missing_all_pct": miss_all[n] * 100, "missing_train_val_pct": miss_tv[n] * 100,
                     "train_zero_share": norm.loc[n, "train_zero_share"], "train_std_m": norm.loc[n, "train_std"],
                     "event_peak_rise_median_m": float(np.median(peaks[n])) if peaks[n] else np.nan,
                     "event_peak_rise_max_m": float(np.max(peaks[n])) if peaks[n] else np.nan,
                     "loso_event_mae_idw": loso.loc[n, "event_mae_idw_p1"], "loso_event_mae_3": loso.loc[n, "event_mae_ridge_rain_time"],
                     "gain_4_vs_3": loso.loc[n, "gain_ridge_rain_time_nbr_vs_3"], "gain_5_vs_3": loso.loc[n, "gain_hgb_rain_time_nbr_vs_3"],
                     **nn[n]})
    return pd.DataFrame(rows).set_index("sensor")


def main() -> None:
    t = build_table()
    t.to_csv(OUT_CSV, encoding="utf-8-sig", float_format="%.4f")
    others = [n for n in t.index if n not in HARM]
    num = ["dist_to_road_node_m", "gauge_dist_m", "rain_corr_best", "missing_all_pct", "missing_train_val_pct", "train_zero_share",
           "train_std_m", "event_peak_rise_median_m", "nn_road_m", "nn3_mean_road_m", "nn_corr", "max_corr", "median_corr",
           "gain_4_vs_3", "gain_5_vs_3"]
    L = []
    w = L.append
    w("[강남] 이웃 정보가 해가 된 센서 4개 정리 (사실만, 원인 판정 없음)")
    w("=" * 74)
    w("대상: " + ", ".join(HARM) + " (T2 LOSO에서 (4) 또는 (5)가 (3)보다 이벤트 MAE가 나빠진 센서)")
    for n in HARM:
        r = t.loc[n]
        w("")
        w(f"[{n}] {r.district}, 주소 '{r.address}'" + (f" / 시설명 '{r.place}'" if isinstance(r.place, str) else ""))
        w(f"  좌표 ({r.lat:.5f}, {r.lon:.5f}), 좌표 근거 {r.location_basis}, 시설 위치 대용 {r.facility_proxy}, 검토 필요 표시 {r.needs_review}, 도로노드까지 {r.dist_to_road_node_m:.0f} m")
        w(f"  최근접 이웃 센서 {r.nn_id}: 도로 {r.nn_road_m:.0f} m (직선 {r.nn_straight_m:.0f} m), 변화량 상관 {r.nn_corr:.2f}; 최근접 3개 평균 도로거리 {r.nn3_mean_road_m:.0f} m")
        w(f"  다른 센서와의 변화량 상관: 중앙값 {r.median_corr:.2f}, 최대 {r.max_corr:.2f} ({r.max_corr_with}, 도로 {r.max_corr_road_m:.0f} m)")
        w(f"  결측률: 3개월 {r.missing_all_pct:.2f}%, train+val {r.missing_train_val_pct:.2f}%")
        w(f"  수위 특성: train 0 m 비율 {r.train_zero_share:.1%}, train 표준편차 {r.train_std_m:.4f} m, 이벤트 첨두 상승(중앙값) {r.event_peak_rise_median_m:.3f} m (최대 {r.event_peak_rise_max_m:.3f} m)")
        w(f"  강우 관계: 배정 강우량계 {r.gauge}({r.gauge_gu}) {r.gauge_dist_m:.0f} m, 수위 변화량과 강우 최대 상관 {r.rain_corr_best:.2f} (시차 {int(r.rain_corr_lag_min)}분)")
        w(f"  LOSO 이벤트 MAE: IDW {r.loso_event_mae_idw:.4f} m, (3) {r.loso_event_mae_3:.4f} m; (4) 개선율 {r.gain_4_vs_3:+.1%}, (5) 개선율 {r.gain_5_vs_3:+.1%}")
    w("")
    w("나머지 14개 센서의 중앙값 (맥락)")
    for c in num:
        w(f"  {c:26s} {t.loc[others, c].median():9.4f}   (이 4개 센서 중앙값 {t.loc[HARM, c].median():9.4f})")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
