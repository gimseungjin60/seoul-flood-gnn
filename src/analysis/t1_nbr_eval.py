"""G-none·재현 판정: 규칙 src/analysis/t1_nbr_rules.py 의 N1·N2·R1만 판정하고 나머지는 참고로 보고한다. 결과를 보기 전에 작성(2026-11-14).

실행: python -m src.analysis.t1_nbr_eval   (docs/28 예측과 t1_nbr 예측이 모두 있어야 한다)
출력: data/processed/gangnam2026_t1_nbr_report.txt
유효 표본은 docs/28 판정과 같다(τ 실제와 M0 예측이 모두 있음).
"""
import numpy as np
import pandas as pd

from src.analysis.t1_long import OUT_PRED as LONG_PRED
from src.analysis.t1_long_rules import HORIZONS
from src.analysis.t1_nbr import OUT_PRED as NBR_PRED
from src.analysis.t1_nbr_rules import FREEZE_DATE, MAIN_H, n1_verdict, n2_verdict, r1_verdict
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_nbr_report.txt"
MODELS = ("persist", "ar_rain", "gnn_road", "gnn_corr", "gnn_none", "gnn_road_s345")
NEW = {"gnn_none", "gnn_road_s345"}
LABEL = {"persist": "M0 직전 값", "ar_rain": "M2 자기 이력+강우", "gnn_road": "G-road (시드 0·1·2)", "gnn_corr": "G-corr",
         "gnn_none": "G-none 이웃 없음", "gnn_road_s345": "G-road (시드 3·4·5)"}


def event_error_sums(pred: dict, actual: np.ndarray, ev: list, models) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(actual) & np.isfinite(pred["persist"])
    for m in models:
        if (ok & ~np.isfinite(pred[m])).any():
            raise ValueError(f"{m}: 유효 표본에 예측이 없다")
    S, N = np.zeros((len(ev), actual.shape[1], len(models))), np.zeros((len(ev), actual.shape[1]))
    for e, (_, idx) in enumerate(ev):
        o = ok[idx]
        N[e] = o.sum(0)
        for i, m in enumerate(models):
            S[e, :, i] = np.where(o, np.abs(pred[m][idx] - actual[idx]), 0.0).sum(0)
    return S, N


def main() -> None:
    z1, z2 = np.load(LONG_PRED, allow_pickle=True), np.load(NBR_PRED, allow_pickle=True)
    if not (np.array_equal(z1["bt"], z2["bt"]) and np.array_equal(z1["nodes"], z2["nodes"])):
        raise ValueError("docs/28 예측과 t1_nbr 예측의 시각·센서가 다르다")
    bt = pd.DatetimeIndex(z1["bt"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    ev = event_blocks(bt, events)
    actual = z1["actual_m"].astype(float)
    k = {m: i for i, m in enumerate(MODELS)}
    L = [f"이웃 없는 GNN·시드 재현 (규칙 고정 {FREEZE_DATE}, docs/30). 입력 120분. 서울시 강남구 2026 예측 대상 {actual.shape[1]}개, "
         f"이벤트 MAE(mm), 이벤트 {len(ev)}개 복원추출 {N_BOOT}회", ""]
    for h in sorted(HORIZONS, reverse=True):
        pred = {m: (z2 if m in NEW else z1)[f"pred_{m}_h{h}"].astype(float) for m in MODELS}
        S, N = event_error_sums(pred, actual, ev, MODELS)
        point, _ = macro_mae(S, N, np.ones((1, len(ev))))
        boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
        main = h == MAIN_H
        L.append(f"[{'판정' if main else '참고'}: {h * 10}분 뒤] 유효 표본 {int(N.sum())}")
        for m in MODELS:
            lo, hi = ci(boot[:, k[m]])
            L.append(f"  {LABEL[m]:<20} {point[0, k[m]] * 1000:7.2f} [{lo * 1000:.2f}, {hi * 1000:.2f}]")
        for a, b, tag, fn in (("gnn_road", "gnn_none", "N1", n1_verdict), ("gnn_none", "ar_rain", "N2", n2_verdict),
                              ("gnn_road_s345", "ar_rain", "R1", r1_verdict), ("gnn_corr", "gnn_none", "참고", None),
                              ("gnn_road_s345", "gnn_road", "참고", None)):
            lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
            d = point[0, k[a]] - point[0, k[b]]
            judged = main and fn is not None
            L.append(f"  {tag if main else '참고'} {LABEL[a]} − {LABEL[b]}: {d * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]"
                     + (f"  → {fn(lo, hi)}" if judged else ""))
        L.append("")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
