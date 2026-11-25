"""T1 확장(입력 120분) 판정: 규칙 src/analysis/t1_long_rules.py 의 D1·D2만 판정하고 나머지는 참고로 보고한다. 결과를 보기 전에 작성(2026-11-14).

실행: python -m src.analysis.t1_long_eval   (먼저 python -m src.analysis.t1_long)
출력: data/processed/gangnam2026_t1_long_report.txt
"""
import numpy as np
import pandas as pd

from src.analysis.t1_long import OUT_PRED
from src.analysis.t1_long_rules import FREEZE_DATE, HORIZONS, MAIN_H
from src.analysis.t1_st_eval import error_sums
from src.analysis.t1_st_rules import LABEL, MODELS, f_verdict
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_long_report.txt"


def main() -> None:
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    k = {m: i for i, m in enumerate(MODELS)}
    L = [f"T1 확장: 입력 120분 (규칙 고정 {FREEZE_DATE}, docs/26). 서울시 강남구 2026 예측 대상 {z['actual_m'].shape[1]}개, "
         f"이벤트 MAE(mm), 이벤트 {len(events)}개 복원추출 {N_BOOT}회", ""]
    for h in sorted(HORIZONS, reverse=True):
        S, N, *_ = error_sums(z, h, bt, events)
        point, _ = macro_mae(S, N, np.ones((1, S.shape[0])))
        boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
        main = h == MAIN_H
        L.append(f"[{'판정' if main else '참고'}: {h * 10}분 뒤] 유효 표본 {int(N.sum())}")
        for m in MODELS:
            lo, hi = ci(boot[:, k[m]])
            L.append(f"  {LABEL[m]:<16} {point[0, k[m]] * 1000:7.1f} [{lo * 1000:.1f}, {hi * 1000:.1f}]")
        for a, b, tag in (("gnn_road", "ar_rain", "D1"), ("gnn_corr", "ar_rain", "D2"), ("ar_rain_nbr", "ar_rain", "참고"),
                          ("ar_rain", "persist", "참고")):
            lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
            d = point[0, k[a]] - point[0, k[b]]
            judged = main and tag in ("D1", "D2")
            L.append(f"  {tag if main else '참고'} {LABEL[a]} − {LABEL[b]}: {d * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]"
                     + (f"  → {f_verdict(lo, hi)}" if judged else ""))
        L.append("")
    L.append("GNN 그래프 2개(G-road, G-corr)를 비교했다(다중 비교 보정 없음). T1(입력 60분)과의 비교는 다른 실행이라 수치만 나란히 본다.")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
