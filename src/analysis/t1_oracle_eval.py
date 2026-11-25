"""미래 강우 상한 판정: 규칙 src/analysis/t1_oracle_rules.py 의 O1·O2만 판정하고 나머지는 참고로 보고한다. 결과를 보기 전에 작성(2026-11-14).

실행: python -m src.analysis.t1_oracle_eval   (먼저 python -m src.analysis.t1_oracle)
출력: data/processed/gangnam2026_t1_oracle_report.txt
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.analysis import t1_warning
from src.analysis.t1_nbr_eval import event_error_sums
from src.analysis.t1_oracle import OUT_PRED
from src.analysis.t1_oracle_rules import FREEZE_DATE, MAIN_H, THRESH, o1_verdict, o2_verdict
from src.analysis.t1_st import OUT_PRED as T1_PRED
from src.analysis.t1_st_rules import HORIZONS
from src.analysis.t1_warning_rules import REF_THRESH, csi
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_oracle_report.txt"
MODELS = ("persist", "ar_rain", "ar_rain_fut", "ar_rain_nbr_fut")
LABEL = {"persist": "M0 직전 값", "ar_rain": "M2 자기 이력+강우", "ar_rain_fut": "M2F +미래 강우", "ar_rain_nbr_fut": "M3F +이웃+미래 강우"}


@contextmanager
def warning_models(models):
    old = t1_warning.MODELS
    t1_warning.MODELS = models
    try:
        yield
    finally:
        t1_warning.MODELS = old


def main() -> None:
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    ev = event_blocks(bt, events)
    actual = z["actual_m"].astype(float)
    k = {m: i for i, m in enumerate(MODELS)}
    L = [f"미래 강우 상한 실험 (규칙 고정 {FREEZE_DATE}, docs/32). 실시간에는 쓸 수 없는 상한. 서울시 강남구 2026 예측 대상 {actual.shape[1]}개, "
         f"이벤트 {len(ev)}개 복원추출 {N_BOOT}회", ""]
    if T1_PRED.exists():
        t1 = np.load(T1_PRED, allow_pickle=True)
        d = max(np.nanmax(np.abs(t1[f"pred_ar_rain_h{h}"].astype(float) - z[f"pred_ar_rain_h{h}"].astype(float))) for h in HORIZONS)
        L += [f"[확인] 다시 만든 M2 와 T1 M2 의 최대 차이 {d * 1000:.4f} mm", ""]
    for h in sorted(HORIZONS, reverse=True):
        pred = {m: z[f"pred_{m}_h{h}"].astype(float) for m in MODELS}
        S, N = event_error_sums(pred, actual, ev, MODELS)
        point, _ = macro_mae(S, N, np.ones((1, len(ev))))
        boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
        main = h == MAIN_H
        L.append(f"[{'판정' if main else '참고'}: {h * 10}분 뒤, 이벤트 MAE] 유효 표본 {int(N.sum())}")
        for m in MODELS:
            lo, hi = ci(boot[:, k[m]])
            L.append(f"  {LABEL[m]:<18} {point[0, k[m]] * 1000:7.2f} [{lo * 1000:.2f}, {hi * 1000:.2f}]")
        for a, b, tag in (("ar_rain_fut", "ar_rain", "O1"), ("ar_rain_nbr_fut", "ar_rain", "참고")):
            lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
            dd = point[0, k[a]] - point[0, k[b]]
            L.append(f"  {tag if main else '참고'} {LABEL[a]} − {LABEL[b]}: {dd * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]"
                     + (f"  → {o1_verdict(lo, hi)}" if main and tag == "O1" else ""))
        ok = np.isfinite(actual) & np.isfinite(pred["persist"])
        inev = np.zeros(len(bt), bool)
        for _, idx in ev:
            inev[idx] = True
        sel = ok & inev[:, None] & ((actual - pred["persist"]) * 1000 >= 200)
        L.append(f"  참고 실제 {h * 10}분 상승 200 mm 이상 표본 {int(sel.sum())}개의 평균 절대오차(mm): "
                 + ", ".join(f"{LABEL[m].split()[0]} {np.abs(pred[m] - actual)[sel].mean() * 1000:.1f}" for m in MODELS))
        L.append("")
    with warning_models(MODELS):
        for h_blk, thresh, main in [(MAIN_H, THRESH, True), (1, THRESH, False), (MAIN_H, REF_THRESH, False), (1, REF_THRESH, False)]:
            C, n_none = t1_warning.counts(z, h_blk, thresh, bt, events)
            tot = C.sum(0)
            point = np.array([csi(*tot[i]) for i in range(len(MODELS))])
            bs = t1_warning.boot_csi(C)
            L.append(f"[{'판정' if main else '참고'}: {h_blk * 10}분 뒤, {thresh:.0f} mm 첫 도달 감지] 도달 사례 {int(tot[0, 1])}개, "
                     f"비도달 사례 {int(n_none.sum())}개")
            for m in MODELS:
                i = k[m]
                L.append(f"  {LABEL[m]:<18} CSI {point[i]:.3f}  맞힘 {int(tot[i, 0])}/{int(tot[i, 1])}  헛경보 {int(tot[i, 2])}/{int(n_none.sum())}")
            for a, tag in (("ar_rain_fut", "O2"), ("ar_rain_nbr_fut", "참고")):
                lo, hi = np.nanpercentile(bs[:, k[a]] - bs[:, k["ar_rain"]], [2.5, 97.5])
                L.append(f"  {tag if main else '참고'} CSI {LABEL[a]} − M2: {point[k[a]] - point[k['ar_rain']]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                         + (f"  → {o2_verdict(lo, hi)}" if main and tag == "O2" else ""))
            L.append("")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
