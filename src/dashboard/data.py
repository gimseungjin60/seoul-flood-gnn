"""대시보드용 2026 자료 준비. 표시 기준은 docs/16(평소 대비 상승 단계)을 따른다. 새로 계산하는 모델은 없다.

- 상승폭: src/analysis/rise_levels.rise_mm (10분 수위 − 센서별 요일x시 중앙값, mm)
- 단계 색: 평가 대상 + 기록이 안정적인 이웃 센서만. 기록 불안정 센서(rise_levels.unstable_nodes)는 회색.
- 비교선: 평가 대상 17개는 2026 재비교의 LOSO 추정(IDW, GNN)을 함께 보여준다(gangnam2026_t2_{loso,gnn}_predictions.npz, git 제외,
  python -m src.analysis.t2_2026 --part loso / gnn 으로 재생성). 이 예측의 상승폭은 폴드 학습 구간 평균 기준이라 지도 단계와 기저선이 조금 다르다.
"""
import numpy as np
import pandas as pd

from src.analysis.rise_levels import event_mask, rise_mm, stage, unstable_nodes
from src.analysis.spatial_signal_profile_adjusted import hour_of_week
from src.analysis.t2_2026 import OUT_PRED, load_roles
from src.data.assign_rain_gauges import node_blocks
from src.data.build_station_graph_2026 import OUT_NODES
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

    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    _, inev = event_mask(bt, events)
    nbr = [c for c, n in enumerate(nodes) if n not in roles["targets"]]
    unstable = {nodes[c] for c, _ in unstable_nodes(rise, inev, nbr)}

    info = pd.read_csv(OUT_NODES).set_index("node_id").loc[nodes, ["lat", "lon"]].reset_index()
    info["role"] = ["평가 대상" if n in roles["targets"] else "이웃 센서" for n in nodes]
    info["unstable"] = info.node_id.isin(unstable)

    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    asg = pd.read_csv(OUT_GAUGE, dtype={"gauge_id": str}).set_index("node_id")
    pos = rt.get_indexer(bt)
    rain = np.stack([z["values"][pos, gids.index(asg.loc[n, "gauge_id"])].astype(float) for n in nodes], axis=1)

    est = {}
    if OUT_PRED["loso"].exists() and OUT_PRED["gnn"].exists():
        lz, gz = np.load(OUT_PRED["loso"], allow_pickle=True), np.load(OUT_PRED["gnn"], allow_pickle=True)
        if not np.array_equal(pd.DatetimeIndex(lz["bt"]), bt):
            raise ValueError("예측 파일의 시각이 수위 블록과 다르다")
        for j, n in enumerate(str(x) for x in lz["nodes"]):
            est[n] = pd.DataFrame({"실측(같은 기준)": lz["actual_m"][:, j], "IDW 추정": lz["pred_idw_p1"][:, j],
                                   "GNN 추정": gz["pred_gnn"][:, j]}, index=bt).astype(float) * 1000
    return dict(nodes=nodes, info=info, bt=bt, rise=rise, stage=np.where(np.isfinite(rise), stage(np.nan_to_num(rise)), -1),
                rain=rain, events=events, est=est)
