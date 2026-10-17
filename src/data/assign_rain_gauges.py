"""센서(18개) → 최근접 강우량계 배정 + '센서별 최근접 강우'가 '강남 평균 강우'보다 수위 반응을 더 잘 설명하는지 확인.

실행: python -m src.data.assign_rain_gauges
입력: data/raw/서울시 강우량계 위치정보.xlsx (48개소 위경도), rainfall_10min_2025.npz, gangnam_station_timeseries.npz,
      gangnam_station_graph_nodes.csv
출력: data/processed/gangnam_station_rain_gauge.csv, rainfall_gauge_assignment_report.txt

배정 규칙(사전 고정): 센서 좌표 - 강우량계 좌표의 직선거리(haversine)가 가장 가까운 1개소. 구 경계는 보지 않는다(비는 구 경계를 따르지 않음).
검증: 센서별 10분 평균 수위의 변화량과 강우의 교차상관(강우가 앞서는 시차 0~30분)의 최댓값을
      (a) 최근접 강우량계 (b) 강남 3개소 평균 으로 비교한다. train+val 구간만 사용(test 미사용).
"""
import time

import numpy as np
import pandas as pd

from src.data.clean_nodes import PROCESSED
from src.data.inspect_water_level import PROJECT_ROOT
from src.data.prepare_rainfall import OUT_NPZ as RAIN_NPZ, VAL_END
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ

LOC_XLSX = PROJECT_ROOT / "data" / "raw" / "서울시 강우량계 위치정보.xlsx"
NODES_CSV = PROCESSED / "gangnam_station_graph_nodes.csv"
OUT_CSV = PROCESSED / "gangnam_station_rain_gauge.csv"
REPORT = PROCESSED / "rainfall_gauge_assignment_report.txt"
R_EARTH = 6371008.8
MAX_LAG = 3          # 10분 단위, 0~30분


def read_locations(path=LOC_XLSX) -> pd.DataFrame:
    d = pd.read_excel(path, dtype=str)
    d = d.rename(columns={"구청명": "gu", "강우량계명": "gname", "주소": "addr", "강우량계 코드": "gid", "경도": "lon", "위도": "lat"})
    d["lon"], d["lat"] = d["lon"].astype(float), d["lat"].astype(float)
    return d[["gu", "gname", "addr", "gid", "lat", "lon"]]


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """(N,) x (M,) → (N, M) 미터."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(h))


def assign_nearest(nodes: pd.DataFrame, loc: pd.DataFrame) -> pd.DataFrame:
    D = haversine_matrix(nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), loc["lat"].to_numpy(), loc["lon"].to_numpy())
    order = np.argsort(D, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    r = np.arange(len(nodes))
    return pd.DataFrame({"node_id": nodes["node_id"].to_numpy(), "gauge_id": loc["gid"].to_numpy()[i1],
                         "gauge_name": loc["gname"].to_numpy()[i1], "gauge_gu": loc["gu"].to_numpy()[i1],
                         "dist_m": D[r, i1].round(0), "second_gauge_id": loc["gid"].to_numpy()[i2],
                         "second_gauge_name": loc["gname"].to_numpy()[i2], "second_dist_m": D[r, i2].round(0)})


def node_blocks(values: np.ndarray, mask: np.ndarray, min_obs: int = 5) -> np.ndarray:
    """(T, N) 1분 → (T//10, N) 10분 평균(관측 min_obs개 미만이면 NaN). 블록 끝 라벨 :09 (강우 격자와 같은 구간)."""
    v = np.where(mask > 0, values, np.nan).astype(np.float64)
    t = v.shape[0] // 10 * 10
    x = v[:t].reshape(-1, 10, v.shape[1])
    cnt = np.isfinite(x).sum(axis=1)
    return np.where(cnt >= min_obs, np.where(np.isfinite(x), x, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)


def best_lag_corr(rain: np.ndarray, dy: np.ndarray, max_lag: int = MAX_LAG) -> tuple[float, int]:
    best = (np.nan, 0)
    for k in range(max_lag + 1):
        a, b = rain[:len(rain) - k], dy[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 200 or a[ok].std() == 0 or b[ok].std() == 0:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if not np.isfinite(best[0]) or c > best[0]:
            best = (c, k * 10)
    return best


def main() -> None:
    t0 = time.time()
    loc = read_locations()
    nodes = pd.read_csv(NODES_CSV).sort_values("idx").reset_index(drop=True)
    z = np.load(RAIN_NPZ, allow_pickle=True)
    gids = [str(g) for g in z["gauge_ids"]]
    assert set(gids) == set(loc["gid"]), "강우 CSV와 위치 파일의 강우량계가 다르다"
    asg = assign_nearest(nodes, loc)
    asg.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    rt = pd.DatetimeIndex(z["times"])
    s = np.load(SEWER_NPZ, allow_pickle=True)
    s_ids = [str(n) for n in s["node_ids"]]
    assert s_ids == nodes["node_id"].tolist(), "센서 순서가 그래프 노드 순서와 다르다"
    st = pd.DatetimeIndex(s["times"])
    blk = node_blocks(s["values"][:VAL_END], s["mask"][:VAL_END])
    bt = st[:len(blk) * 10][9::10]
    rain = pd.DataFrame(z["values"], index=rt, columns=gids).reindex(bt)            # NaN 유지
    gn = [g for g, u in zip(gids, z["gauge_gu"]) if u == "강남구"]
    mean_gn = rain[gn].mean(axis=1, skipna=True).where(rain[gn].notna().any(axis=1)).to_numpy()
    dy = np.vstack([np.full((1, blk.shape[1]), np.nan), np.diff(blk, axis=0)])
    rows = []
    for i, r in asg.iterrows():
        c_n, l_n = best_lag_corr(rain[r.gauge_id].to_numpy(), dy[:, i])
        c_m, l_m = best_lag_corr(mean_gn, dy[:, i])
        rows.append({"node_id": r.node_id, "gauge": r.gauge_name, "dist_m": r.dist_m, "corr_nearest": c_n, "lag_nearest": l_n,
                     "corr_gangnam_mean": c_m, "lag_mean": l_m})
    cmp_ = pd.DataFrame(rows)
    cmp_["nearest_better"] = cmp_.corr_nearest > cmp_.corr_gangnam_mean

    L = []
    w = L.append
    w("[강우량계 배정] 센서 18개 → 최근접 강우량계 (원본 수정 없음, test 구간 미사용)")
    w("=" * 70)
    w(f"위치 파일: 강우량계 {len(loc)}개소 (강우 CSV와 코드 일치 {set(gids) == set(loc['gid'])}), 좌표 범위 위도 {loc.lat.min():.4f}~{loc.lat.max():.4f}, "
      f"경도 {loc.lon.min():.4f}~{loc.lon.max():.4f}")
    w("")
    w("1. 배정 (직선거리 최근접 1개소)")
    w(asg.to_string(index=False))
    w(f"  지점별 사용 횟수: {asg.gauge_name.value_counts().to_dict()}")
    w(f"  거리: 최소 {asg.dist_m.min():.0f} m, 중앙값 {asg.dist_m.median():.0f} m, 최대 {asg.dist_m.max():.0f} m "
      f"(2위와의 거리 차 중앙값 {(asg.second_dist_m - asg.dist_m).median():.0f} m)")
    cross = asg[asg.gauge_gu != "강남구"]
    w(f"  강남 밖 강우량계가 최근접인 센서 {len(cross)}개: " + ", ".join(f"{r.node_id}({r.gauge_name} {r.dist_m:.0f}m)" for r in cross.itertuples()))
    w("")
    w("2. 검증: 센서 수위 10분 변화량과의 최대 교차상관 (강우 선행 0~30분, train+val)")
    w(cmp_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    w(f"  최근접이 강남 평균보다 높은 센서 {int(cmp_.nearest_better.sum())}/{len(cmp_)}, "
      f"상관 평균 최근접 {cmp_.corr_nearest.mean():.3f} vs 강남 평균 {cmp_.corr_gangnam_mean.mean():.3f}")
    w("")
    w("주의")
    w("  - 센서 좌표는 주소/시설 기준이라 수십 m 오차가 있다. 강우량계까지 거리가 수백 m~2.4 km라 이 오차는 배정에 영향이 작다.")
    w("  - 변화량 상관은 센서마다 반응 크기가 달라 절대 비교가 아니다. 두 강우 입력을 같은 센서에서 비교한 것이다.")
    w("  - 23-0014(0 m 90%)처럼 변화가 거의 없는 센서는 상관이 불안정하다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
