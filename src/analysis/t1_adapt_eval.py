"""G-adapt 판정: 규칙 src/analysis/t1_adapt_rules.py 의 A1·A2만 판정하고 나머지는 참고로 보고한다. 결과를 보기 전에 작성(2026-11-14).

실행: python -m src.analysis.t1_adapt_eval   (T1 예측과 G-adapt 예측이 모두 있어야 한다)
출력: data/processed/gangnam2026_t1_adapt_report.txt
유효 표본은 T1 판정과 같다(τ 실제와 M0 예측이 모두 있음). 모든 비교 모델이 그 표본에서 값이 있어야 한다.
"""
import numpy as np
import pandas as pd

from src.analysis.t1_adapt import OUT_ADJ, OUT_PRED as ADAPT_PRED
from src.analysis.t1_adapt_rules import A2_NO, A2_YES, A2_ZERO, FREEZE_DATE
from src.analysis.t1_st import OUT_PRED as T1_PRED
from src.analysis.t1_st_rules import HORIZONS, MAIN_H, f_verdict
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_adapt_report.txt"
MODELS = ("persist", "ar_rain", "ar_rain_nbr", "gnn_road", "gnn_corr", "gnn_adapt")
LABEL = {"persist": "M0 직전 값", "ar_rain": "M2 자기 이력+강우", "ar_rain_nbr": "M3 +이웃", "gnn_road": "G-road GNN",
         "gnn_corr": "G-corr GNN", "gnn_adapt": "G-adapt GNN"}


def a2_verdict(lo: float, hi: float) -> str:
    return A2_YES if hi < 0 else (A2_NO if lo > 0 else A2_ZERO)


def main() -> None:
    z1, z2 = np.load(T1_PRED, allow_pickle=True), np.load(ADAPT_PRED, allow_pickle=True)
    if not (np.array_equal(z1["bt"], z2["bt"]) and np.array_equal(z1["nodes"], z2["nodes"])):
        raise ValueError("T1 예측과 G-adapt 예측의 시각·센서가 다르다")
    bt = pd.DatetimeIndex(z1["bt"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    ev = event_blocks(bt, events)
    actual = z1["actual_m"].astype(float)
    k = {m: i for i, m in enumerate(MODELS)}
    L = [f"G-adapt: 연결을 학습하는 시공간 GNN (규칙 고정 {FREEZE_DATE}, docs/27). 서울시 강남구 2026 예측 대상 {actual.shape[1]}개, "
         f"이벤트 MAE(mm), 이벤트 {len(ev)}개 복원추출 {N_BOOT}회", ""]
    for h in sorted(HORIZONS, reverse=True):
        pred = {m: (z2 if m == "gnn_adapt" else z1)[f"pred_{m}_h{h}"].astype(float) for m in MODELS}
        ok = np.isfinite(actual) & np.isfinite(pred["persist"])
        for m in MODELS:
            if (ok & ~np.isfinite(pred[m])).any():
                raise ValueError(f"h={h} {m}: 유효 표본에 예측이 없다")
        S, N = np.zeros((len(ev), actual.shape[1], len(MODELS))), np.zeros((len(ev), actual.shape[1]))
        for e, (_, idx) in enumerate(ev):
            o = ok[idx]
            N[e] = o.sum(0)
            for i, m in enumerate(MODELS):
                S[e, :, i] = np.where(o, np.abs(pred[m][idx] - actual[idx]), 0.0).sum(0)
        point, _ = macro_mae(S, N, np.ones((1, len(ev))))
        boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
        main = h == MAIN_H
        L.append(f"[{'판정' if main else '참고'}: {h * 10}분 뒤] 유효 표본 {int(N.sum())}")
        for m in MODELS:
            lo, hi = ci(boot[:, k[m]])
            L.append(f"  {LABEL[m]:<14} {point[0, k[m]] * 1000:7.2f} [{lo * 1000:.2f}, {hi * 1000:.2f}]")
        for a, b, tag, fn in (("gnn_adapt", "ar_rain", "A1", f_verdict), ("gnn_adapt", "gnn_road", "A2", a2_verdict)):
            lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
            d = point[0, k[a]] - point[0, k[b]]
            L.append(f"  {tag if main else '참고'} {LABEL[a]} − {LABEL[b]}: {d * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]"
                     + (f"  → {fn(lo, hi)}" if main else ""))
        L.append("")
    if OUT_ADJ.exists():
        A = pd.read_csv(OUT_ADJ, index_col=0)
        top = A.stack().sort_values(ascending=False).head(10)
        L.append("[참고: 폴드 1·시드 0 학습된 연결 점수 상위 10 (받는 센서 ← 보내는 센서)]")
        L += [f"  {r} ← {c}  {v:.3f}" for (r, c), v in top.items()]
        L.append(f"  점수 0인 연결 비율 {(A.stack() == 0).mean():.1%}")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
