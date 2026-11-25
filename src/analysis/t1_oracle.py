"""미래 강우 상한 실험 실행: 규칙 src/analysis/t1_oracle_rules.py (docs/32). T1 설정(입력 60분, 10·30분 뒤), Ridge 만.
t1_st.run_fold 를 graphs=() 로 부르고, 강우 특징에 미래 강우 열만 붙인다. 예측 거리마다 따로 한 번씩 돌린다.

실행: python -m src.analysis.t1_oracle
출력: data/processed/gangnam2026_t1_oracle_predictions.npz (재생성 가능, git 제외)
"""
import time

import numpy as np
import pandas as pd

from src.analysis import t1_st
from src.analysis.t1_st_rules import HORIZONS
from src.analysis.t2_2026 import build_inputs, load_roles
from src.analysis.t2_loso import N_FOLDS, fold_slices
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_GAUGE, OUT_NPZ as RAIN_NPZ

OUT_PRED = PROCESSED / "gangnam2026_t1_oracle_predictions.npz"
NAMES = ("ar_rain", "ar_rain_fut", "ar_rain_nbr_fut")


def future_rain_features(rain: np.ndarray, pos: np.ndarray, h: int) -> np.ndarray:
    """rain (S,) mm/10분(NaN=결측), pos (B,) 블록 t 의 슬롯. 블록 t+1..t+h 강우의 log1p (B, h). 격자 밖·결측은 0."""
    r0 = np.nan_to_num(rain, nan=0.0)
    cols = []
    for j in range(1, h + 1):
        idx = pos + j
        cols.append(np.where(idx < len(r0), r0[np.minimum(idx, len(r0) - 1)], 0.0))
    return np.log1p(np.stack(cols, axis=1))


def node_rain(nodes: list, bt: pd.DatetimeIndex) -> tuple[list, np.ndarray]:
    """build_inputs 와 같은 강우량계 배정으로 센서별 강우 원시 시계열과 블록 슬롯 위치."""
    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    pos = rt.get_indexer(bt)
    asg = pd.read_csv(OUT_GAUGE, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    return [z["values"][:, gids.index(asg.loc[n, "gauge_id"])].astype(float) for n in nodes], pos


def main() -> None:
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    B = inp["blk"].shape[0]
    rains, pos = node_rain(nodes, inp["bt"])
    fut_inp = {h: {**inp, "rain_feat": [np.hstack([rf, future_rain_features(r, pos, h)]) for rf, r in zip(inp["rain_feat"], rains)]}
               for h in HORIZONS}
    print(f"노드 {len(nodes)}개, 예측 대상 {len(tg)}개, 블록 {B:,}개", flush=True)
    store = {n: [np.full((B, len(tg)), np.nan) for _ in HORIZONS] for n in ("persist", *NAMES)}
    t0 = time.time()
    for f, (eval_idx, _, train_mask) in enumerate(fold_slices(B)):
        base, scale, _ = t1_st.run_fold(inp, tg, eval_idx, train_mask, graphs=())
        pn = {"persist": np.zeros_like(base["ar_rain"]), "ar_rain": base["ar_rain"],
              "ar_rain_fut": np.full_like(base["ar_rain"], np.nan), "ar_rain_nbr_fut": np.full_like(base["ar_rain"], np.nan)}
        for hc, h in enumerate(HORIZONS):
            out, scale_h, _ = t1_st.run_fold(fut_inp[h], tg, eval_idx, train_mask, graphs=())
            assert np.allclose(scale, scale_h)
            pn["ar_rain_fut"][:, :, hc] = out["ar_rain"][:, :, hc]          # 그 거리까지의 미래 강우로 학습한 그 거리 예측만
            pn["ar_rain_nbr_fut"][:, :, hc] = out["ar_rain_nbr"][:, :, hc]
        t1_st.to_levels(pn, scale, eval_idx, tg, inp["blk"], store)
        print(f"  fold {f + 1}/{N_FOLDS} ({time.time() - t0:.0f}초)", flush=True)
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array([nodes[j] for j in tg]),
                        horizons=np.array(HORIZONS), actual_m=inp["blk"][:, tg].astype(np.float32),
                        **{f"pred_{n}_h{h}": store[n][hc].astype(np.float32) for n in store for hc, h in enumerate(HORIZONS)})
    print(f"저장: {OUT_PRED.name}. 판정은 python -m src.analysis.t1_oracle_eval", flush=True)


if __name__ == "__main__":
    main()
