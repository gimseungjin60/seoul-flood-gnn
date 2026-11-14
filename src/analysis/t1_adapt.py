"""T1 확장 G-adapt(연결을 학습하는 시공간 GNN) 실행: 규칙 src/analysis/t1_adapt_rules.py (docs/27).
t1_st.run_fold·train_gnn 을 그대로 쓰고 모델 클래스만 AdaptiveSTGNN 으로 바꾼다. Ridge 는 돌리지 않는다(M0~M3·G-road 는 T1 실행 값).

실행: python -m src.analysis.t1_adapt
출력: data/processed/gangnam2026_t1_adapt_predictions.npz (재생성 가능, git 제외), 폴드 1·시드 0 의 학습된 연결 강도 csv
"""
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.analysis import t1_st
from src.analysis.t1_adapt_rules import EMB_DIM, EMB_INIT_SD
from src.analysis.t1_st_rules import HIDDEN, HORIZONS
from src.analysis.t2_2026 import build_inputs, load_roles
from src.analysis.t2_loso import N_FOLDS, fold_slices
from src.data.clean_nodes import PROCESSED

OUT_PRED = PROCESSED / "gangnam2026_t1_adapt_predictions.npz"
OUT_ADJ = PROCESSED / "gangnam2026_t1_adapt_adjacency_fold1_seed0.csv"
_N_NODES = {"n": None}
_LAST = {"model": None}


class AdaptiveSTGNN(t1_st.STGNN):
    """t1_st.STGNN 과 같은 GRU·인코더·메시지 전달 구조. 인접 행렬만 학습되는 센서 벡터에서 만든다(입력 A 는 무시)."""

    def __init__(self, stat_dim: int, hidden: int = HIDDEN, n_layers: int = 2, n_out: int = len(HORIZONS)):
        super().__init__(stat_dim, hidden, n_layers, n_out)
        n = _N_NODES["n"]
        self.e_src = nn.Parameter(torch.randn(n, EMB_DIM) * EMB_INIT_SD)
        self.e_dst = nn.Parameter(torch.randn(n, EMB_DIM) * EMB_INIT_SD)
        _LAST["model"] = self

    def adjacency_scores(self) -> torch.Tensor:
        return torch.relu(self.e_dst @ self.e_src.T)

    def forward(self, seq, stat, m, A=None):
        B, N, W, _ = seq.shape
        _, hT = self.gru(seq.reshape(B * N, W, 2))
        h = torch.relu(self.enc(torch.cat([hT[-1].reshape(B, N, -1), stat], dim=-1)))
        valid = (m[:, None, :] > 0) & ~torch.eye(N, dtype=torch.bool)[None]  # 관측된 이웃만, 자기 자신 제외
        s = self.adjacency_scores()[None].expand(B, N, N).masked_fill(~valid, -1e9)
        w = torch.softmax(s, dim=-1) * valid
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)                 # 관측 이웃이 없으면 0 (NaN 없음)
        for sl, nl in zip(self.self_lin, self.nbr_lin):
            h = torch.relu(sl(h) + nl(w @ h))
        return self.out(h)


@contextmanager
def adaptive_model(n_nodes: int):
    old = t1_st.STGNN
    _N_NODES["n"] = n_nodes
    t1_st.STGNN = AdaptiveSTGNN
    try:
        yield
    finally:
        t1_st.STGNN = old


def main() -> None:
    torch.set_num_threads(1)
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    B = inp["blk"].shape[0]
    print(f"노드 {len(nodes)}개, 예측 대상 {len(tg)}개, 블록 {B:,}개", flush=True)
    store = {"gnn_adapt": [np.full((B, len(tg)), np.nan) for _ in HORIZONS]}
    t0 = time.time()
    with adaptive_model(len(nodes)):
        for f, (eval_idx, _, train_mask) in enumerate(fold_slices(B)):
            tf = time.time()
            pn, scale, _ = t1_st.run_fold(inp, tg, eval_idx, train_mask, graphs=("road",), ridge=False)
            pn = {"gnn_adapt": pn["gnn_road"]}                         # 키 이름만 road, 모델은 AdaptiveSTGNN (A 무시)
            t1_st.to_levels(pn, scale, eval_idx, tg, inp["blk"], store)
            print(f"  fold {f + 1}/{N_FOLDS} ({time.time() - tf:.0f}초)", flush=True)
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array([nodes[j] for j in tg]),
                        horizons=np.array(HORIZONS), actual_m=inp["blk"][:, tg].astype(np.float32),
                        **{f"pred_gnn_adapt_h{h}": store["gnn_adapt"][hc].astype(np.float32) for hc, h in enumerate(HORIZONS)})
    print(f"저장: {OUT_PRED.name} ({time.time() - t0:.0f}초). 판정은 python -m src.analysis.t1_adapt_eval", flush=True)
    save_reference_adjacency(inp, tg, nodes)


def save_reference_adjacency(inp, tg, nodes) -> None:
    """참고용: 폴드 1, 시드 0 으로 한 번 더 학습해 연결 점수를 저장한다(판정에 쓰지 않음)."""
    B = inp["blk"].shape[0]
    eval_idx, _, train_mask = next(fold_slices(B))
    with adaptive_model(len(nodes)):
        t1_st.run_fold(inp, tg, eval_idx, train_mask, graphs=("road",), seeds=(0,), ridge=False)
        with torch.no_grad():
            s = torch.relu(_LAST["model"].e_dst @ _LAST["model"].e_src.T).numpy()
    np.fill_diagonal(s, np.nan)
    pd.DataFrame(s, index=nodes, columns=nodes).to_csv(OUT_ADJ, encoding="utf-8-sig", float_format="%.4f")
    print(f"저장: {OUT_ADJ.name} (행 = 받는 센서)", flush=True)


if __name__ == "__main__":
    main()
