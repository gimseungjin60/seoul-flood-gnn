"""2026 강우(10분, 2026-05~08) 격자 + 강남 강우 이벤트 + pool 센서 최근접 강우량계 배정. 2025판(prepare_rainfall, assign_rain_gauges)과 같은 규칙.

실행: python -m src.data.prepare_rainfall_2026
입력(읽기만): data/raw/rain/2026/rain_2026{05..08}.csv (cp949), data/raw/rain/gauges/서울시 강우량계 위치정보.xlsx,
             data/processed/gangnam2026_station_nodes.csv (J2 통과 센서 좌표), gangnam2026_station_timeseries.npz(시각 확인용)
출력: data/processed/rainfall_10min_2026.npz, gangnam2026_rain_events.csv, gangnam2026_station_rain_gauge.csv,
      gangnam2026_rainfall_report.txt

- 격자·결측·이벤트 규칙은 2025와 같다(구간 끝 라벨 :09, 결측은 NaN, 강남 3개소 평균, 6시간 무강우 분리, 총량 10 mm 이상).
- 2026-05는 선행 24시간 누적 계산용으로만 쓰고 이벤트는 2026-06-01~08-31만 센다.
- 평가 이벤트 표시(is_eval)는 사전 고정 규칙(t2_2026_rules.eval_events: 끝 + 3시간이 자료 끝 안)대로.
- 강우량계 배정은 직선거리 최근접 1개소(2025와 같음, 구 경계 무시).
"""
import numpy as np
import pandas as pd

from src.analysis.t2_2026_rules import EVENT_AFTER_H, eval_events
from src.data.assign_rain_gauges import assign_nearest, read_locations
from src.data.build_station_graph_2026 import OUT_NODES
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import GANGNAM, SIGNIFICANT_MM, find_events, read_raw, to_grid
from src.data.prepare_station_timeseries_2026 import END, OUT_NPZ as TS_NPZ, START
from src.data.raw_paths import RAIN_2026_CSVS

OUT_NPZ = PROCESSED / "rainfall_10min_2026.npz"
OUT_EVENTS = PROCESSED / "gangnam2026_rain_events.csv"
OUT_GAUGE = PROCESSED / "gangnam2026_station_rain_gauge.csv"
REPORT = PROCESSED / "gangnam2026_rainfall_report.txt"
ARTIFACT_WINDOWS = [("2026-07-07 13:49", "2026-07-07 14:45")]   # 수위 품질 규칙 R2 최대 구간(보고용)


def main() -> None:
    d = read_raw([RAIN_2026_CSVS[m] for m in ("05", "06", "07", "08")])
    off = (d["t_raw"].dt.minute % 10) != 9
    times, gids, info, values = to_grid(d)
    mask = np.isfinite(values).astype(np.int8)
    np.savez_compressed(OUT_NPZ, times=times.to_numpy().astype("datetime64[s]"), gauge_ids=np.array(gids),
                        gauge_names=info["gname"].to_numpy(), gauge_gu=info["gu"].to_numpy(), values=values, mask=mask)

    gn = [i for i, g in enumerate(gids) if info.loc[g, "gu"] == GANGNAM]
    with np.errstate(all="ignore"):
        mean_gn = np.nanmean(values[:, gn], axis=1)
    win = (times >= START) & (times < END)
    ev = find_events(np.where(win, mean_gn, np.nan), times)
    sig = ev[ev.total_mm >= SIGNIFICANT_MM].reset_index(drop=True)
    sig.insert(0, "event_id", range(1, len(sig) + 1))
    data_end = pd.Timestamp(np.load(TS_NPZ)["times"][-1])
    sig["is_eval"] = sig.event_id.isin(eval_events(sig, data_end).event_id)
    sig.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig", float_format="%.3f")

    loc = read_locations()
    assert set(gids) == set(loc["gid"]), "강우 CSV와 위치 파일의 강우량계가 다르다"
    nodes = pd.read_csv(OUT_NODES)
    nodes = nodes[nodes.j2_pass].reset_index(drop=True)
    asg = assign_nearest(nodes, loc)
    asg.to_csv(OUT_GAUGE, index=False, encoding="utf-8-sig")

    L = []
    w = L.append
    w("[서울시 강우량 정보 2026-05~08] 격자·이벤트·강우량계 배정 (원본 수정 없음)")
    w("=" * 70)
    w(f"입력 {len(d):,}행, 강우량계 {len(gids)}개, 10분 격자 {len(times):,}칸 ({times[0]} ~ {times[-1]}, 라벨=구간 끝)")
    w(f"수집 시각이 :09 규칙을 벗어난 행 {int(off.sum())}개 (구간 끝 라벨로 매핑), 음수 {int((values < 0).sum())}개, "
      f"최대 {np.nanmax(values):.1f} mm/10분")
    miss = (mask == 0).sum(axis=0)
    w(f"결측 칸 {int((mask == 0).sum()):,}/{mask.size:,}, 강남 3개소: "
      + ", ".join(f"{info.loc[gids[i], 'gname']} {int(miss[i])}" for i in gn))
    w(f"\n이벤트 (강남 3개소 평균, 6시간 무강우 분리, 총량 {SIGNIFICANT_MM:.0f} mm 이상): 유의 {len(sig)}개 / 전체 {len(ev)}개, "
      f"평가 {int(sig.is_eval.sum())}개, 50 mm 이상(평가) {int((sig.is_eval & (sig.total_mm >= 50)).sum())}개")
    w(sig[["event_id", "start", "end", "total_mm", "max_10min_mm", "duration_h", "is_eval"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    for a, b in ARTIFACT_WINDOWS:
        hit = sig[(sig.start <= pd.Timestamp(b)) & (sig.end + pd.Timedelta(hours=EVENT_AFTER_H) >= pd.Timestamp(a))]
        w(f"수위 이상 구간 {a}~{b[-5:]}과 겹치는 이벤트 창: {hit.event_id.tolist() or '없음'}")
    w(f"\n강우량계 배정 (J2 통과 센서 {len(asg)}개, 직선거리 최근접)")
    w(f"  사용 지점: {asg.gauge_name.value_counts().to_dict()}")
    w(f"  거리 최소 {asg.dist_m.min():.0f} m, 중앙값 {asg.dist_m.median():.0f} m, 최대 {asg.dist_m.max():.0f} m")
    cross = asg[asg.gauge_gu != GANGNAM]
    w(f"  강남 밖 강우량계가 최근접인 센서 {len(cross)}개: "
      + ", ".join(f"{r.node_id}({r.gauge_name} {r.dist_m:.0f}m)" for r in cross.itertuples()))
    used = [gids.index(g) for g in asg.gauge_id.unique()]
    after_gap = np.vstack([np.zeros((1, values.shape[1]), bool), np.isnan(values[:-1])]) & (values >= 20)
    w(f"\n결측 직후 20 mm/10분 이상(끊긴 동안 누적된 값으로 추정, 값은 그대로 둠): 전체 {int(after_gap.sum())}칸, "
      f"배정된 강우량계 {int(after_gap[:, used].sum())}칸. 배정된 강우량계 최대 "
      + ", ".join(f"{info.loc[gids[i], 'gname']} {np.nanmax(values[:, i]):.1f}" for i in used))
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
