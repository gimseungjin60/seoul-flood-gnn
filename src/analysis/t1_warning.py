"""T1 예측의 '큰 상승 첫 도달' 30분 전 감지: 사전 고정 규칙(src/analysis/t1_warning_rules.py, docs/24)대로 W1·W2 판정과 참고 집계.

실행: python -m src.analysis.t1_warning   (먼저 python -m src.analysis.t1_st 가 끝나야 한다)
출력: data/processed/gangnam2026_t1_warning_report.txt, gangnam2026_t1_warning_counts.csv
"""
import numpy as np
import pandas as pd

from src.analysis.spatial_signal_profile_adjusted import hour_of_week, week_profile
from src.analysis.t1_st import OUT_PRED
from src.analysis.t1_st_rules import HORIZONS, LABEL, MODELS
from src.analysis.t1_warning_rules import (FREEZE_DATE, MAIN_H, N_BOOT, REF_THRESH, SEED, THRESH, case_outcome, csi,
                                           w_verdict)
from src.analysis.t2_loso import event_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_warning_report.txt"
OUT_COUNTS = PROCESSED / "gangnam2026_t1_warning_counts.csv"


def counts(z, h: int, thresh: float, bt, events) -> np.ndarray:
    """(이벤트, 모델, 3) = 맞힘, 도달 사례, 헛경보. 비도달 사례 수는 따로 반환."""
    actual = z["actual_m"].astype(float)
    how = hour_of_week(bt - pd.Timedelta(minutes=9))
    prof = week_profile(actual, how, np.ones(len(actual), bool), "median")[how]
    rise = (actual - prof) * 1000
    pred = {m: (z[f"pred_{m}_h{h}"].astype(float) - prof) * 1000 for m in MODELS}
    issue = pred["persist"]                                                   # M0 예측 = 기점 τ−h 실제
    ok = np.isfinite(rise) & np.isfinite(issue)
    ev = event_blocks(bt, events)
    C = np.zeros((len(ev), len(MODELS), 3))
    n_none = np.zeros(len(ev))
    for e, (_, idx) in enumerate(ev):
        for j in range(actual.shape[1]):
            sel = idx[ok[idx, j]]
            for k, m in enumerate(MODELS):
                out = case_outcome(rise[sel, j], pred[m][sel, j], issue[sel, j], thresh)
                if out is None:
                    continue
                kind, flag = out
                if kind == "reach":
                    C[e, k, 0] += flag
                    C[e, k, 1] += 1
                else:
                    C[e, k, 2] += flag
                    n_none[e] += k == 0
    return C, n_none


def boot_csi(C: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    E = C.shape[0]
    out = np.empty((n_boot, C.shape[1]))
    for b in range(n_boot):
        s = C[rng.integers(0, E, E)].sum(0)
        out[b] = [csi(*s[k]) for k in range(C.shape[1])]
    return out


def main() -> None:
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    k = {m: i for i, m in enumerate(MODELS)}
    L = [f"T1 예측의 큰 상승 첫 도달 감지 (규칙 고정 {FREEZE_DATE}, docs/24). 서울시 강남구 2026 예측 대상 {z['actual_m'].shape[1]}개",
         f"CSI = 맞힘 / (도달 사례 + 헛경보). 이벤트 {len(events)}개 복원추출 {N_BOOT}회(시드 {SEED})", ""]
    rows = []
    for h_blk, thresh, main in [(MAIN_H, THRESH, True), (1, THRESH, False), (MAIN_H, REF_THRESH, False), (1, REF_THRESH, False)]:
        C, n_none = counts(z, h_blk, thresh, bt, events)
        tot = C.sum(0)
        point = np.array([csi(*tot[i]) for i in range(len(MODELS))])
        bs = boot_csi(C)
        title = f"{h_blk * 10}분 뒤, {thresh:.0f} mm 이상"
        L.append(f"[{'판정' if main else '참고'}: {title}] 도달 사례 {int(tot[0, 1])}개, 비도달 사례 {int(n_none.sum())}개")
        for m in MODELS:
            i = k[m]
            lo, hi = np.nanpercentile(bs[:, i], [2.5, 97.5])
            hr = tot[i, 0] / tot[i, 1] if tot[i, 1] else np.nan
            fr = tot[i, 2] / n_none.sum() if n_none.sum() else np.nan
            L.append(f"  {LABEL[m]:<16} CSI {point[i]:.3f} [{lo:.3f}, {hi:.3f}]  맞힘 {int(tot[i, 0])}/{int(tot[i, 1])} ({hr:.0%})  "
                     f"헛경보 {int(tot[i, 2])}/{int(n_none.sum())} ({fr:.0%})")
            rows.append(dict(horizon_min=h_blk * 10, thresh_mm=thresh, model=m, hits=tot[i, 0], reach=tot[i, 1],
                             false_alarms=tot[i, 2], n_none=n_none.sum(), csi=point[i], csi_lo=lo, csi_hi=hi))
        for a, b, tag in (("gnn_road", "ar_rain", "W1"), ("gnn_corr", "ar_rain", "W2"), ("ar_rain_nbr", "ar_rain", "참고")):
            d = bs[:, k[a]] - bs[:, k[b]]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            verdict = w_verdict(lo, hi) if (main and tag != "참고") else ""
            L.append(f"  {tag if main else '참고'} {LABEL[a]} − {LABEL[b]}: {point[k[a]] - point[k[b]]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                     + (f"  → {verdict}" if verdict else ""))
        L.append("")
    L.append("GNN 그래프 2개(G-road, G-corr)를 비교했다(다중 비교 보정 없음). 상승 단계는 위험 판정이 아니다.")
    pd.DataFrame(rows).to_csv(OUT_COUNTS, index=False, encoding="utf-8-sig", float_format="%.4f")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
