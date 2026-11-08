"""2026 그래프 보완 실험 실행: 사전 고정 규칙(src/analysis/t2_graph_rules.py, docs/14)대로 G1(고도 방향)·G2(랜덤 대조군) GNN 예측을 만든다.
판정은 python -m src.analysis.t2_graph_eval 에서만 한다. 그래프 말고는 2026 재비교(t2_2026 --part gnn)와 전부 같다.

실행: python -m src.data.sensor_elevation_2026          # 센서 고도 (G1 전에)
      python -m src.analysis.t2_graph --part diag      # 모델 실행 전 진단: G1 대체 노드 수, 도로거리 이웃과 겹치는 정도
      python -m src.analysis.t2_graph --part elev      # G1 (17 타깃 x 5폴드 x 3시드)
      python -m src.analysis.t2_graph --part rand      # G2 (그래프 시드 = GNN 시드 0·1·2)
출력: data/processed/gangnam2026_t2_graph_diag.txt, gangnam2026_t2_graph_{elev,rand}_predictions.npz (재생성 가능, git 제외)
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.analysis.t2_2026 import build_inputs, load_roles, run_gnn, save
from src.analysis.t2_graph_rules import K, elev_neighbors, nearest, rand_neighbors, random_scores, weights_from
from src.analysis.t2_loso import N_FOLDS
from src.analysis.t2_gnn import SEEDS
from src.data.clean_nodes import PROCESSED
from src.data.sensor_elevation_2026 import OUT_ELEV

OUT_PRED = {g: PROCESSED / f"gangnam2026_t2_graph_{g}_predictions.npz" for g in ("elev", "rand")}
OUT_DIAG = PROCESSED / "gangnam2026_t2_graph_diag.txt"


def node_elevation(nodes: list[str]) -> np.ndarray:
    e = pd.read_csv(OUT_ELEV).set_index("node_id")
    missing = [n for n in nodes if n not in e.index]
    if missing:
        raise ValueError(f"고도가 없는 센서: {missing}")
    return e.loc[nodes, "elev_m"].to_numpy(float)


def make_adjacency(dist: np.ndarray, pick_for_seed):
    """pick_for_seed(seed) -> pick(r, cols). 반환: t2_gnn.gnn_loso_predict 의 adjacency(others, target, seed)."""
    def adjacency(others, target, seed):
        pick = pick_for_seed(seed)
        n = len(others)
        A = np.zeros((n + (target is not None), n + (target is not None)))
        A[:n, :n] = weights_from(dist, others, others, lambda r: pick(r, others))
        if target is not None:
            A[n, :n] = weights_from(dist, [target], others, lambda r: pick(r, others))[0]
        return A
    return adjacency


def elev_adjacency(dist, elev):
    return make_adjacency(dist, lambda _seed: (lambda r, cols: elev_neighbors(dist, elev, r, cols)[0]))


def rand_adjacency(dist):
    scores = {}

    def pick_for_seed(seed):
        if seed not in scores:
            scores[seed] = random_scores(len(dist), seed)
        return lambda r, cols: rand_neighbors(scores[seed], r, cols)
    return make_adjacency(dist, pick_for_seed)


def diagnostics(inp: dict, roles: dict) -> str:
    nodes, dist = inp["nodes"], inp["dist"]
    elev = node_elevation(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    L = ["[강남 2026] 그래프 보완 실험 실행 전 진단 (규칙: src/analysis/t2_graph_rules.py, docs/14). 모델 결과 없음.",
         f"노드 {len(nodes)}개, 타깃 {len(tg)}개, k={K}. 고도 = Copernicus GLO-30 3x3 최솟값(m).",
         f"고도: 최소 {elev.min():.1f}, 중앙 {np.median(elev):.1f}, 최대 {elev.max():.1f} m",
         "", "[G1] 평가 시 타깃 행(타깃을 뺀 47개 후보): 도로거리 5개 이웃 중 타깃보다 높은 이웃 수"]
    n_fb, n_up = 0, []
    for i in tg:
        others = [j for j in range(len(nodes)) if j != i]
        near = nearest(dist, i, others)
        up, fb = elev_neighbors(dist, elev, i, others)
        n_fb += fb
        n_up.append(0 if fb else len(up))
        L.append(f"  {nodes[i]} 고도 {elev[i]:5.1f} m: 높은 이웃 {0 if fb else len(up)}/{len(near)}" + (" → 대체(5개 그대로)" if fb else ""))
    L.append(f"  대체 타깃 {n_fb}/{len(tg)}, 남는 이웃 수 평균 {np.mean([u if u else K for u in n_up]):.2f}")
    allc = list(range(len(nodes)))
    fb_all = sum(elev_neighbors(dist, elev, r, allc)[1] for r in allc)
    L.append(f"  참고: 48개 전체 노드끼리(타깃 제외 전)일 때 대체 노드 {fb_all}/{len(nodes)}")
    L += ["", "[G2] 랜덤 이웃 5개 중 도로거리 5개 이웃과 겹치는 수(48개 노드 평균), 이웃까지 도로거리 중앙값"]
    near_d = np.median([dist[r, allc[b]] for r in allc for b in nearest(dist, r, allc)])
    L.append(f"  도로거리 그래프: 이웃 거리 중앙값 {near_d:.0f} m")
    for g in SEEDS:
        s = random_scores(len(nodes), g)
        ov = np.mean([len(set(rand_neighbors(s, r, allc)) & set(nearest(dist, r, allc))) for r in allc])
        d = np.median([dist[r, allc[b]] for r in allc for b in rand_neighbors(s, r, allc)])
        L.append(f"  그래프 시드 {g}: 겹침 평균 {ov:.2f}/{K}, 이웃 거리 중앙값 {d:.0f} m")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["diag", "elev", "rand"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda s: print(s, flush=True)
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    if args.part == "diag":
        txt = diagnostics(inp, roles)
        OUT_DIAG.write_text(txt, encoding="utf-8")
        print(txt)
        return
    adj = elev_adjacency(inp["dist"], node_elevation(nodes)) if args.part == "elev" else rand_adjacency(inp["dist"])
    log(f"G={args.part}: 노드 {len(nodes)}개, 타깃 {len(tg)}개, {len(tg) * N_FOLDS * len(SEEDS)}회")
    t0 = time.time()
    pred, act = run_gnn(inp, tg, tg, log, adjacency=adj)
    save(OUT_PRED[args.part], inp, tg, act, {"gnn": pred})
    log(f"저장: {OUT_PRED[args.part].name} ({time.time() - t0:.0f}초). 판정은 python -m src.analysis.t2_graph_eval")


if __name__ == "__main__":
    main()
