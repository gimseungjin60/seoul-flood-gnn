"""대시보드용 2026 자료 준비. 표시 기준은 docs/16(평소 대비 상승 단계)을 따른다. 새로 계산하는 모델은 없다.

- 상승폭: src/analysis/rise_levels.rise_mm (10분 수위 − 센서별 요일x시 중앙값, mm)
- 단계 색: 평가 대상 + 기록이 안정적인 이웃 센서만. 기록 불안정 센서(rise_levels.unstable_nodes)는 회색.
- 강남 평균 강우: 강남 강우량계 평균(10분). 사건 총강우량과 같은 기준(prepare_rainfall_2026)이라 '지금까지 내린 비' 누적에 쓴다.
- 30분 뒤 예상: 시공간 예측(gangnam2026_t1_st_predictions.npz, python -m src.analysis.t1_st)의 M2(강우+자기이력 선형, docs/19). 각 시각 τ 의 값은
  τ−30분에 낸 예측이다. GNN은 선형과 구분되지 않아(docs/19) 쓰지 않는다. 예측은 절대 수위(m)라서 지도와 같은 기저선(실측 수위 − 상승폭)을 빼서 상승폭으로 바꿨다.
"""
import numpy as np
import pandas as pd

from src.analysis.rise_levels import event_mask, rise_mm, stage, unstable_nodes
from src.analysis.spatial_signal_profile_adjusted import hour_of_week, week_profile
from src.analysis.t1_st import OUT_PRED as T1_PRED
from src.analysis.flood_overlay import OUT_OVERLAY as OUT_FLOOD
from src.analysis.rain_vs_rise import by_event
from src.analysis.t2_2026 import load_roles
from src.data.assign_rain_gauges import node_blocks
from src.data.build_station_graph_2026 import OUT_NODES
from src.data.prepare_rainfall import GANGNAM
from src.data.prepare_rainfall_2026 import OUT_EVENTS, OUT_GAUGE, OUT_NPZ as RAIN_NPZ
from src.data.prepare_station_timeseries_2026 import OUT_NPZ as TS_NPZ


def load() -> dict:
    roles = load_roles()
    nodes = roles["pool"]
    s = np.load(TS_NPZ)
    ids = [str(x) for x in s["node_ids"]]
    col = [ids.index(n) for n in nodes]
    blk = node_blocks(s["values"][:, col], s["mask"][:, col])
    bt = pd.DatetimeIndex(s["times"])[: len(blk) * 10][9::10]
    how = hour_of_week(bt - pd.Timedelta(minutes=9))
    rise = rise_mm({"blk": blk, "how": how})
    prof = week_profile(blk, how, np.ones(len(blk), bool), "median")    # (168, 센서) 평소 수위(m), 실시간 모드의 기준

    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    _, inev = event_mask(bt, events)
    nbr = [c for c, n in enumerate(nodes) if n not in roles["targets"]]
    unstable = {nodes[c] for c, _ in unstable_nodes(rise, inev, nbr)}

    info = pd.read_csv(OUT_NODES).set_index("node_id").loc[nodes, ["lat", "lon"]].reset_index()
    info["role"] = ["평가 대상" if n in roles["targets"] else "이웃 센서" for n in nodes]
    info["unstable"] = info.node_id.isin(unstable)
    info["flood_txt"] = ""                                                    # 침수예상도 위치 관계(docs/22), 파일이 있을 때만
    if OUT_FLOOD.exists():
        ov = pd.read_csv(OUT_FLOOD).set_index("node_id")
        info["flood_txt"] = [(" · 침수예상도: 영역 안(침수심1)" if ov.at[n, "flood_depth_class"] >= 1 else
                              f" · 침수예상도: 영역 밖(가까운 면 {int(ov.at[n, 'dist_to_flood_m'])} m)") if n in ov.index else "" for n in info.node_id]

    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    asg = pd.read_csv(OUT_GAUGE, dtype={"gauge_id": str}).set_index("node_id")
    pos = rt.get_indexer(bt)
    rain = np.stack([z["values"][pos, gids.index(asg.loc[n, "gauge_id"])].astype(float) for n in nodes], axis=1)

    gn = [i for i, g in enumerate(z["gauge_gu"]) if g == GANGNAM]
    with np.errstate(all="ignore"):
        rain_gn = np.nanmean(z["values"][pos][:, gn], axis=1)

    fc = {}
    if T1_PRED.exists():
        tz = np.load(T1_PRED, allow_pickle=True)
        if not np.array_equal(pd.DatetimeIndex(tz["bt"]), bt):
            raise ValueError("시공간 예측 파일의 시각이 수위 블록과 다르다")
        for j, n in enumerate(str(x) for x in tz["nodes"]):
            c = nodes.index(n)
            lvl = tz["actual_m"][:, j].astype(float) * 1000
            base = lvl - rise[:, c]                    # 지도와 같은 기저선(요일x시 중앙값) = 실측 수위 − 상승폭
            fc[n] = pd.Series(tz["pred_ar_rain_h3"][:, j].astype(float) * 1000 - base, index=bt, name="30분 전 예측(선형)")
    return dict(nodes=nodes, info=info, bt=bt, rise=rise, stage=np.where(np.isfinite(rise), stage(np.nan_to_num(rise)), -1),
                prof=prof, rain=rain, rain_gn=rain_gn, events=events, fc=fc,
                history=by_event(bt, events, rise, info.unstable.to_numpy()))
