"""T1 오차가 어디서 나는가 (사후 참고 분석, 판정에 쓰지 않음, 2026-11-14 T1 판정 후 작성).

질문: GNN은 평균 오차가 근소하게 작지만(docs/19) 큰 상승 감지는 선형보다 덜 잡았다(docs/25). 실제 변화 크기별로 나누면 설명되는가.
- 표본: T1 판정과 같은 이벤트 창 안 유효 표본(τ 실제와 M0 예측이 모두 있음).
- 실제 변화 = 실제 수위(τ) − 실제 수위(τ − h) (mm). 크기 구간별로 모델별 평균 절대오차와 평균 부호 오차(예측 − 실제)를 본다.
- 구간 경계(mm): |변화| < 10, 10~50, 50~200, ≥ 200 과 상승/하강 구분. 경계는 측정 단위 10 mm와 상승 단계(50/200 mm)에서 가져왔다.
- 표본 가중 평균이며 센서·이벤트 매크로 평균이 아니다(판정 지표와 다름). 부트스트랩 없음.

실행: python -m src.analysis.t1_error_breakdown          (T1, 입력 60분)
      python -m src.analysis.t1_error_breakdown --long   (확장, 입력 120분, docs/28 참고 분석)
출력: data/processed/gangnam2026_t1_error_breakdown.txt / gangnam2026_t1_long_error_breakdown.txt
"""
import sys

import numpy as np
import pandas as pd

from src.analysis.t1_st import OUT_PRED
from src.analysis.t1_st_rules import HORIZONS, LABEL
from src.analysis.t2_loso import event_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT = PROCESSED / "gangnam2026_t1_error_breakdown.txt"
SHOW = ("persist", "ar_rain", "ar_rain_nbr", "gnn_road", "gnn_corr")
BINS = [(-np.inf, -200), (-200, -50), (-50, -10), (-10, 10), (10, 50), (50, 200), (200, np.inf)]
NAMES = ["하강 ≥200", "하강 50~200", "하강 10~50", "변화 <10", "상승 10~50", "상승 50~200", "상승 ≥200"]


def main(pred_path=OUT_PRED, horizons=HORIZONS, out=OUT) -> None:
    z = np.load(pred_path, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    ev = ev[ev.is_eval].reset_index(drop=True)
    inev = np.zeros(len(bt), bool)
    for _, idx in event_blocks(bt, ev):
        inev[idx] = True
    act = z["actual_m"].astype(float)
    L = ["T1 오차 분해 (사후 참고, 판정에 쓰지 않음). 서울시 강남구 2026 예측 대상 17개, 이벤트 창 안 유효 표본, 표본 가중 평균",
         "각 칸: 평균 절대오차 mm / 평균 부호 오차(예측 − 실제) mm", ""]
    for h in sorted(horizons, reverse=True):
        pred = {m: z[f"pred_{m}_h{h}"].astype(float) for m in SHOW}
        ok = np.isfinite(act) & np.isfinite(pred["persist"]) & inev[:, None]
        dchg = (act - pred["persist"])[ok] * 1000                    # 실제 변화 = τ 실제 − τ−h 실제
        L.append(f"[{h * 10}분 뒤] 유효 표본 {int(ok.sum()):,}")
        L.append("  " + f"{'실제 변화(mm)':<14}{'표본':>7}{'비율':>7}  " + "  ".join(f"{LABEL[m].split()[0]:>14}" for m in SHOW))
        for (lo, hi), nm in zip(BINS, NAMES):
            sel = (dchg >= lo) & (dchg < hi)
            if not sel.any():
                continue
            cells = []
            for m in SHOW:
                e = (pred[m][ok] - act[ok])[sel] * 1000
                cells.append(f"{np.abs(e).mean():6.1f}/{e.mean():+6.1f}")
            L.append(f"  {nm:<14}{int(sel.sum()):>7}{sel.mean():>7.1%}  " + "  ".join(f"{c:>14}" for c in cells))
        tot = []
        for m in SHOW:
            e = (pred[m][ok] - act[ok]) * 1000
            tot.append(f"{np.abs(e).mean():6.1f}/{e.mean():+6.1f}")
        L.append(f"  {'전체':<14}{int(ok.sum()):>7}{1:>7.0%}  " + "  ".join(f"{c:>14}" for c in tot))
        L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--long" in sys.argv:
        from src.analysis.t1_long_rules import HORIZONS as LONG_H
        main(PROCESSED / "gangnam2026_t1_long_predictions.npz", LONG_H,
             PROCESSED / "gangnam2026_t1_long_error_breakdown.txt")
    else:
        main()
