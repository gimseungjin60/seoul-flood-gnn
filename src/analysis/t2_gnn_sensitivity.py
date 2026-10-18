"""사후 민감도 분석 (사후, 판정에 쓰지 않음): T2 GNN 하이퍼파라미터 그리드 k∈{3,5} × 마스크 비율 p∈{0.2,0.4} × 이벤트 가중 w∈{1,5}, 8개 조합.

판정(src/analysis/t2_gnn_eval.py)이 끝난 뒤에만 실행된다(판정 보고서가 없으면 중단). 사전 고정한 t2_gnn.py 는 수정하지 않고, 실행 중에만 모듈 값을 덮어쓴다.
시드는 조합마다 0 하나(판정 실행은 시드 3개 평균이라 단일 시드는 더 흔들린다). 참고로 사전 고정 설정(k=5, p=0.3, w=5)을 같은 방식(시드 0)으로 한 번 더 돌려
시드 하나의 변동 수준을 보여 준다(8개 조합에는 포함하지 않는다).
보고는 8개 설정의 GNN − IDW 점추정 범위와 95% CI 범위(하한·상한의 최솟값~최댓값)로만 한다. 특정 설정을 골라 강조하지 않는다.

실행: python -m src.analysis.t2_gnn_sensitivity
출력: data/processed/gangnam_t2_gnn_sensitivity_report.txt, gangnam_t2_gnn_sensitivity.csv (예측 npz 는 git 제외)
"""
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from src.analysis import t2_gnn
from src.analysis.t2_gnn import BOOT_SEED, N_BOOT, OUT_REPORT as VERDICT_REPORT
from src.analysis.t2_gnn_eval import NAMES, error_sums
from src.analysis.t2_loso import OUT_PRED as LOSO_PRED, build_inputs
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_EVENTS

GRID = list(itertools.product((3, 5), (0.2, 0.4), (1.0, 5.0)))        # (k, p, w)
FROZEN = (5, 0.3, 5.0)                                                 # 사전 고정 설정 (참고용 단일 시드 실행)
SEED = 0
OUT_REPORT = PROCESSED / "gangnam_t2_gnn_sensitivity_report.txt"
OUT_CSV = PROCESSED / "gangnam_t2_gnn_sensitivity.csv"


def tag(cfg) -> str:
    k, p, w = cfg
    return f"k{k}_p{p}_w{int(w)}"


def pred_path(cfg):
    return PROCESSED / f"gangnam_t2_gnn_sens_{tag(cfg)}.npz"


def run_config(cfg) -> str:
    """한 조합을 시드 0 으로 실행하고 예측을 저장한다(프로세스 안에서만 모듈 값을 덮어쓴다)."""
    k, p, w = cfg
    torch.set_num_threads(1)
    orig_knn, orig_ev = t2_gnn.knn_weights, t2_gnn.event_weight_vector
    t2_gnn.knn_weights = lambda dist, rows, cols, k_=None: orig_knn(dist, rows, cols, k)
    t2_gnn.event_weight_vector = lambda bt, events, w_=None: orig_ev(bt, events, w)
    t2_gnn.MASK_RATIO = p
    try:
        inp = build_inputs()
        pred, act = t2_gnn.run_gnn_loso(inp, None, seeds=(SEED,))
    finally:
        t2_gnn.knn_weights, t2_gnn.event_weight_vector = orig_knn, orig_ev
    np.savez_compressed(pred_path(cfg), bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array(inp["nodes"]),
                        actual_m=act.astype(np.float32), pred_gnn=pred.astype(np.float32))
    return tag(cfg)


def evaluate(cfg, z, events) -> dict:
    g = np.load(pred_path(cfg), allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    pred = {m: z[f"pred_{m}"].astype(float) for m in NAMES if m != "gnn"}
    pred["gnn"] = g["pred_gnn"].astype(float)
    S, N, ev_ids = error_sums(pred, z["actual_m"].astype(float), bt, events)
    k = {m: i for i, m in enumerate(NAMES)}
    point, _ = macro_mae(S, N, np.ones((1, S.shape[0])))
    boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
    d = boot[:, k["gnn"]] - boot[:, k["idw_p1"]]
    lo, hi = ci(d)
    d4 = boot[:, k["gnn"]] - boot[:, k["ridge_rain_time_nbr"]]
    lo4, hi4 = ci(d4)
    return {"config": tag(cfg), "k": cfg[0], "mask_p": cfg[1], "event_w": cfg[2], "gnn_mae_mm": point[0][k["gnn"]] * 1000,
            "diff_vs_idw_mm": (point[0][k["gnn"]] - point[0][k["idw_p1"]]) * 1000, "ci_lo": lo * 1000, "ci_hi": hi * 1000,
            "diff_vs_nbr4_mm": (point[0][k["gnn"]] - point[0][k["ridge_rain_time_nbr"]]) * 1000, "ci4_lo": lo4 * 1000, "ci4_hi": hi4 * 1000}


def main() -> None:
    if not VERDICT_REPORT.exists():
        raise SystemExit("판정 보고서(gangnam_t2_gnn_report.txt)가 없다. 민감도 분석은 판정 후에만 실행한다.")
    t0 = time.time()
    cfgs = GRID + [FROZEN]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for done in ex.map(run_config, cfgs):
            print(f"  완료 {done} ({time.time() - t0:.0f}초)", flush=True)
    z = np.load(LOSO_PRED, allow_pickle=True)
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    rows = [evaluate(c, z, events) for c in cfgs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.3f")
    g8, ref = df.iloc[:len(GRID)], df.iloc[len(GRID)]
    L = ["[사후, 판정에 쓰지 않음] T2 GNN 민감도 분석: 8개 조합, 시드 0 하나씩. 판정은 gangnam_t2_gnn_report.txt 의 사전 고정 설정(시드 3개 평균)뿐이다.",
         "특정 설정을 골라 강조하지 않는다. 아래는 범위만 보고한다. GNN − IDW 는 센서 18개 평균 이벤트 MAE 차이(mm), CI 는 이벤트 8개 2000회 복원추출 95%.", "",
         f"{'설정':<16}{'GNN MAE':>9}{'GNN−IDW':>10}{'95% CI':>20}{'GNN−(4)':>10}", "-" * 66]
    for r in g8.itertuples():
        L.append(f"{r.config:<16}{r.gnn_mae_mm:9.2f}{r.diff_vs_idw_mm:+10.2f}   [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}]{r.diff_vs_nbr4_mm:+10.2f}")
    L += ["", "[범위 요약, 8개 설정]",
          f"  GNN − IDW 점추정: {g8.diff_vs_idw_mm.min():+.2f} ~ {g8.diff_vs_idw_mm.max():+.2f} mm",
          f"  95% CI 하한: {g8.ci_lo.min():+.2f} ~ {g8.ci_lo.max():+.2f} mm,  상한: {g8.ci_hi.min():+.2f} ~ {g8.ci_hi.max():+.2f} mm",
          f"  GNN − (4) 점추정: {g8.diff_vs_nbr4_mm.min():+.2f} ~ {g8.diff_vs_nbr4_mm.max():+.2f} mm",
          "", "[참고] 사전 고정 설정을 같은 방식(시드 0 하나)으로 다시 돌린 값(시드 변동 수준을 보기 위한 것, 8개 범위에는 미포함):",
          f"  {ref.config}: GNN MAE {ref.gnn_mae_mm:.2f}, GNN − IDW {ref.diff_vs_idw_mm:+.2f} mm [{ref.ci_lo:+.2f}, {ref.ci_hi:+.2f}] (판정 실행은 시드 3개 평균)"]
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
