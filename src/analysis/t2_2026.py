"""2026 우기 재비교 실행: 사전 고정 규칙(src/analysis/t2_2026_rules.py, docs/11)대로 LOSO 기준 모델 (1)~(5)와 GNN 예측을 만든다.
판정은 python -m src.analysis.t2_2026_eval 에서만 한다. 하이퍼파라미터·모델 정의는 2025 코드(t2_loso, t2_gnn)를 그대로 쓴다.

실행: python -m src.analysis.t2_2026 --part loso          # 주 분석: 기준 모델 (1)~(5), 타깃 17개 x 5폴드
      python -m src.analysis.t2_2026 --part gnn --timing  # GNN 1회(타깃 1 x 폴드 1 x 시드 1) 시간 측정
      python -m src.analysis.t2_2026 --part gnn           # 주 분석 GNN (17 x 5 x 3시드)
      python -m src.analysis.t2_2026 --part h             # H 실험: H를 학습에서 완전히 빼고 H 3개 추정(기준 모델 + GNN)
출력(재생성 가능, git 제외): data/processed/gangnam2026_t2_{loso,gnn,h}_predictions.npz

노드 = pool(J1·J2 적용 후). 타깃 = core(J2 통과). 가짜 타깃·GNN 손실 = 타깃 i 를 뺀 core(labels). 이웃 후보 = i 를 뺀 pool 전체.
H 실험: 노드 = pool − (H − {h}), labels = core − H. h 는 타깃이라 맨 먼저 제거되므로 H는 어떤 학습에도 들어가지 않는다.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.analysis.spatial_signal_profile_adjusted import hour_of_week
from src.analysis.t2_2026_rules import jump_excluded, sensor_roles
from src.analysis.t2_gnn import EPOCHS, SEEDS, event_weight_vector, gnn_loso_predict
from src.analysis.t2_loso import MODELS, N_FOLDS, fold_slices, loso_predict, normalized_rise, nowcast_rain_features, time_features
from src.data.assign_rain_gauges import node_blocks
from src.data.build_station_graph_2026 import OUT_NODES, OUT_PAIRS
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS, OUT_GAUGE, OUT_NPZ as RAIN_NPZ
from src.data.prepare_station_timeseries_2026 import OUT_NPZ as TS_NPZ, OUT_SELECTION

OUT_PRED = {p: PROCESSED / f"gangnam2026_t2_{p}_predictions.npz" for p in ("loso", "gnn", "h")}


def load_roles() -> dict:
    sel = pd.read_csv(OUT_SELECTION)
    z = np.load(TS_NPZ)
    nodes = pd.read_csv(OUT_NODES)
    no_coord = nodes.loc[~nodes.j2_pass, "node_id"].tolist() + sorted(set(sel.loc[sel.in_pool, "node_id"]) - set(nodes.node_id))
    return sensor_roles(sel, jump_excluded(z["values"], list(z["node_ids"])), no_coord)


def h_setup(roles: dict, h: str) -> tuple[list[str], int, list[int]]:
    """H 실험 하나: (노드 목록, 타깃 h 인덱스, 라벨 인덱스). 다른 H 는 노드에서 빠지고, 라벨에는 H 가 하나도 없다."""
    H = set(roles["H"])
    nodes = [n for n in roles["pool"] if n not in H or n == h]
    labels = [nodes.index(n) for n in roles["targets"] if n not in H]
    return nodes, nodes.index(h), labels


def build_inputs(node_ids: list[str]) -> dict:
    """node_ids 순서의 10분 블록·강우 특징·시간대·도로거리·평가 이벤트. 수위는 node_ids 열만 읽는다."""
    s = np.load(TS_NPZ)
    ids = [str(x) for x in s["node_ids"]]
    col = [ids.index(n) for n in node_ids]
    blk = node_blocks(s["values"][:, col], s["mask"][:, col])
    bt = pd.DatetimeIndex(s["times"])[: len(blk) * 10][9::10]
    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    pos = rt.get_indexer(bt)
    if (pos < 0).any():
        raise ValueError("수위 블록 라벨이 강우 격자에 없다")
    asg = pd.read_csv(OUT_GAUGE, dtype={"gauge_id": str}).set_index("node_id").loc[node_ids]
    rain_feat = [nowcast_rain_features(z["values"][:, gids.index(asg.loc[n, "gauge_id"])].astype(float), pos) for n in node_ids]
    nid = {k: c for c, k in enumerate(node_ids)}
    dist = np.zeros((len(node_ids), len(node_ids)))
    for r in pd.read_csv(OUT_PAIRS).itertuples():
        if r.node_a in nid and r.node_b in nid:
            # 같은 도로 노드로 연결된 쌍(도로거리 0, 현재 이웃 전용 23-0088·0089 하나)은 직선거리로 대체 (docs/11, 실행 전 추가)
            d = r.road_dist_m if r.road_dist_m > 0 else r.straight_dist_m
            dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = d
    off = ~np.eye(len(node_ids), dtype=bool)
    if not (dist[off] > 0).all():
        raise ValueError("도로거리가 없거나 0인 센서 쌍이 있다")
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    ev = ev[ev.is_eval].reset_index(drop=True)
    return dict(nodes=node_ids, blk=blk, bt=bt, rain_feat=rain_feat, time_feat=time_features(bt - pd.Timedelta(minutes=9)),
                how=hour_of_week(bt - pd.Timedelta(minutes=9)), dist=dist, events=ev)


def run_baselines(inp: dict, targets: list[int], labels: list[int], log=print):
    B, N = inp["blk"].shape
    pred = {m: np.full((B, N), np.nan) for m in MODELS}
    act = np.full((B, N), np.nan)
    for f, (eval_idx, train_idx, train_mask) in enumerate(fold_slices(B)):
        R, scale = normalized_rise(inp["blk"], inp["how"], train_mask)
        t0 = time.time()
        for i in targets:
            out = loso_predict(R, inp["dist"], inp["rain_feat"], inp["time_feat"], i, train_idx, eval_idx, labels=labels)
            for m in MODELS:
                pred[m][eval_idx, i] = out[m] * scale[i]
            act[eval_idx, i] = R[eval_idx, i] * scale[i]
        log(f"  기준 모델 fold {f + 1}/{N_FOLDS} ({time.time() - t0:.0f}초)")
    return pred, act


def run_gnn(inp: dict, targets: list[int], labels: list[int], log=print, folds=None, seeds=SEEDS):
    B, N = inp["blk"].shape
    pred = np.full((B, N), np.nan)
    act = np.full((B, N), np.nan)
    wv = event_weight_vector(inp["bt"], inp["events"])
    for f, (eval_idx, train_idx, train_mask) in enumerate(fold_slices(B)):
        if folds is not None and f not in folds:
            continue
        R, scale = normalized_rise(inp["blk"], inp["how"], train_mask)
        t0 = time.time()
        for i in targets:
            p = gnn_loso_predict(R, inp["dist"], inp["rain_feat"], inp["time_feat"], i, train_idx, eval_idx, wv, seeds, EPOCHS,
                                 labels=labels)
            pred[eval_idx, i] = p * scale[i]
            act[eval_idx, i] = R[eval_idx, i] * scale[i]
        log(f"  GNN fold {f + 1}/{N_FOLDS} ({time.time() - t0:.0f}초)")
    return pred, act


def save(path, inp, cols, act, preds: dict):
    """타깃 열만 저장. cols = 저장할 노드 인덱스(inp['nodes'] 기준)."""
    np.savez_compressed(path, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array([inp["nodes"][c] for c in cols]),
                        actual_m=act[:, cols].astype(np.float32), **{f"pred_{m}": p[:, cols].astype(np.float32) for m, p in preds.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["loso", "gnn", "h"], required=True)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda s: print(s, flush=True)
    roles = load_roles()
    t0 = time.time()
    if args.part in ("loso", "gnn"):
        nodes = roles["pool"]
        inp = build_inputs(nodes)
        tg = [nodes.index(n) for n in roles["targets"]]
        log(f"노드 {len(nodes)}개, 타깃 {len(tg)}개, 블록 {len(inp['bt']):,}개, 평가 이벤트 {len(inp['events'])}개")
        if args.part == "loso":
            pred, act = run_baselines(inp, tg, tg, log)
            save(OUT_PRED["loso"], inp, tg, act, pred)
        elif args.timing:
            run_gnn(inp, tg[:1], tg, log, folds=[0], seeds=(0,))
            dt = time.time() - t0
            log(f"1회 {dt:.1f}초 → 주 분석 {len(tg) * N_FOLDS * len(SEEDS)}회 예상 {dt * len(tg) * N_FOLDS * len(SEEDS) / 3600:.2f}시간")
            return
        else:
            pred, act = run_gnn(inp, tg, tg, log)
            save(OUT_PRED["gnn"], inp, tg, act, {"gnn": pred})
    else:
        H = roles["H"]
        out_act, out_pred, bt = {}, {}, None
        for h in H:
            nodes, i, labels = h_setup(roles, h)
            inp = build_inputs(nodes)
            log(f"H {h}: 노드 {len(nodes)}개, 라벨 {len(labels)}개")
            pred, act = run_baselines(inp, [i], labels, log)
            g, _ = run_gnn(inp, [i], labels, log)
            pred["gnn"] = g
            out_act[h] = act[:, i]
            for m, p in pred.items():
                out_pred.setdefault(m, {})[h] = p[:, i]
            bt = inp["bt"]
        np.savez_compressed(OUT_PRED["h"], bt=bt.to_numpy().astype("datetime64[s]"), nodes=np.array(H),
                            actual_m=np.stack([out_act[h] for h in H], 1).astype(np.float32),
                            **{f"pred_{m}": np.stack([v[h] for h in H], 1).astype(np.float32) for m, v in out_pred.items()})
    log(f"저장 완료 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
