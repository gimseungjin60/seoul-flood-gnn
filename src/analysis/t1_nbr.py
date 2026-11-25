"""이웃 없는 GNN(G-none)과 G-road 시드 3·4·5 실행: 규칙 src/analysis/t1_nbr_rules.py (docs/30). 설정은 docs/26 확장(입력 120분)과 같다.
t1_st.run_fold 를 그대로 쓰고, G-none 은 t1_st.road_graph 만 0 행렬로 바꾼다. Ridge 는 돌리지 않는다(M2 는 docs/28 실행 값).

실행: python -m src.analysis.t1_nbr
출력: data/processed/gangnam2026_t1_nbr_predictions.npz (재생성 가능, git 제외)
"""
import time
from contextlib import contextmanager

import numpy as np
import torch

from src.analysis import t1_st
from src.analysis.t1_long import long_settings
from src.analysis.t1_long_rules import HORIZONS
from src.analysis.t1_nbr_rules import REPRO_SEEDS
from src.analysis.t1_st_rules import SEEDS
from src.analysis.t2_2026 import build_inputs, load_roles
from src.analysis.t2_loso import N_FOLDS, fold_slices
from src.data.clean_nodes import PROCESSED

OUT_PRED = PROCESSED / "gangnam2026_t1_nbr_predictions.npz"


@contextmanager
def no_neighbors():
    """road_graph → 0 행렬. 메시지 가중치가 모두 0이 되어 이웃 항이 사라진다."""
    old = t1_st.road_graph
    t1_st.road_graph = lambda dist: np.zeros((len(dist), len(dist)))
    try:
        yield
    finally:
        t1_st.road_graph = old


def main() -> None:
    torch.set_num_threads(1)
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    B = inp["blk"].shape[0]
    print(f"노드 {len(nodes)}개, 예측 대상 {len(tg)}개, 블록 {B:,}개", flush=True)
    names = ("gnn_none", "gnn_road_s345")
    t0 = time.time()
    with long_settings():
        store = {n: [np.full((B, len(tg)), np.nan) for _ in HORIZONS] for n in names}
        for f, (eval_idx, _, train_mask) in enumerate(fold_slices(B)):
            tf = time.time()
            with no_neighbors():
                p0, scale, _ = t1_st.run_fold(inp, tg, eval_idx, train_mask, graphs=("road",), seeds=SEEDS, ridge=False)
            p1, scale1, _ = t1_st.run_fold(inp, tg, eval_idx, train_mask, graphs=("road",), seeds=REPRO_SEEDS, ridge=False)
            assert np.allclose(scale, scale1)
            t1_st.to_levels({"gnn_none": p0["gnn_road"], "gnn_road_s345": p1["gnn_road"]}, scale, eval_idx, tg, inp["blk"], store)
            print(f"  fold {f + 1}/{N_FOLDS} ({time.time() - tf:.0f}초)", flush=True)
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array([nodes[j] for j in tg]),
                        horizons=np.array(HORIZONS), actual_m=inp["blk"][:, tg].astype(np.float32),
                        **{f"pred_{n}_h{h}": store[n][hc].astype(np.float32) for n in names for hc, h in enumerate(HORIZONS)})
    print(f"저장: {OUT_PRED.name} ({time.time() - t0:.0f}초). 판정은 python -m src.analysis.t1_nbr_eval", flush=True)


if __name__ == "__main__":
    main()
