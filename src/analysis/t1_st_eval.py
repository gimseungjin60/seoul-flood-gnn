"""T1 시공간 GNN 단기 예측 판정: 사전 고정 규칙(src/analysis/t1_st_rules.py, docs/17)의 F1~F5만 판정하고 나머지는 참고로 보고한다.
결과를 보기 전에 작성했다(2026-11-09).

실행: python -m src.analysis.t1_st_eval   (먼저 python -m src.analysis.t1_st 가 끝나야 한다)
출력: data/processed/gangnam2026_t1_st_report.txt, gangnam2026_t1_st_ci.csv
유효 표본 = 목표 시각 τ 의 실제 수위와 기점 τ − h 의 수위(= M0 예측)가 모두 있는 (τ, 센서). 모든 모델이 이 표본에서 값이 있어야 한다.
"""
import numpy as np
import pandas as pd

from src.analysis.t1_st import OUT_PRED
from src.analysis.t1_st_rules import HORIZONS, LABEL, MAIN_H, MODELS, f3_graph_effect, f5_verdict, f_verdict
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t1_st_report.txt"
OUT_CI = PROCESSED / "gangnam2026_t1_st_ci.csv"


def error_sums(z, h: int, bt, events):
    actual = z["actual_m"].astype(float)
    pred = {m: z[f"pred_{m}_h{h}"].astype(float) for m in MODELS}
    ok = np.isfinite(actual) & np.isfinite(pred["persist"])
    for m in MODELS:
        bad = ok & ~np.isfinite(pred[m])
        if bad.any():
            raise ValueError(f"h={h} {m}: 유효 표본 {int(bad.sum())}개에 예측이 없다")
    ev = event_blocks(bt, events)
    J = actual.shape[1]
    S, N = np.zeros((len(ev), J, len(MODELS))), np.zeros((len(ev), J))
    for e, (_, idx) in enumerate(ev):
        o = ok[idx]
        N[e] = o.sum(0)
        for k, m in enumerate(MODELS):
            S[e, :, k] = np.where(o, np.abs(pred[m][idx] - actual[idx]), 0.0).sum(0)
    return S, N, [i for i, _ in ev], pred, actual, ok


def main() -> None:
    z = np.load(OUT_PRED, allow_pickle=True)
    bt, nodes = pd.DatetimeIndex(z["bt"]), [str(n) for n in z["nodes"]]
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    k = {m: i for i, m in enumerate(MODELS)}
    res, rows = {}, []
    for h in HORIZONS:
        S, N, ev_ids, pred, actual, ok = error_sums(z, h, bt, events)
        point, mae_j = macro_mae(S, N, np.ones((1, S.shape[0])))
        boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)

        def diff(a, b, point=point, boot=boot):
            lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
            return float(point[0, k[a]] - point[0, k[b]]), lo, hi
        res[h] = dict(point=point[0], mae_j=mae_j[0], boot=boot, diff=diff, ev_ids=ev_ids, n=int(N.sum()),
                      all_mae={m: float(np.abs(pred[m][ok] - actual[ok]).mean()) for m in MODELS})
        for m in MODELS:
            lo, hi = ci(boot[:, k[m]])
            rows.append({"stat": f"h{h}_mae_{m}", "point_m": point[0, k[m]], "lo_m": lo, "hi_m": hi})

    H, h10 = MAIN_H, HORIZONS[0]
    f1, f2 = res[H]["diff"]("gnn_road", "ar_rain"), res[H]["diff"]("gnn_corr", "ar_rain")
    v1, v2 = f_verdict(*f1[1:]), f_verdict(*f2[1:])
    f3r, f3c = res[H]["diff"]("gnn_road", "ar_rain_nbr"), res[H]["diff"]("gnn_corr", "ar_rain_nbr")
    g3r, g3c = f3_graph_effect(v1, f3r[2]), f3_graph_effect(v2, f3c[2])
    f4r, f4c = res[h10]["diff"]("gnn_road", "ar_rain"), res[h10]["diff"]("gnn_corr", "ar_rain")
    f5 = res[H]["diff"]("ar_rain", "persist")
    for s, d in (("F1_h3_road_minus_ar_rain", f1), ("F2_h3_corr_minus_ar_rain", f2), ("F3_h3_road_minus_ar_rain_nbr", f3r),
                 ("F3_h3_corr_minus_ar_rain_nbr", f3c), ("F4_h1_road_minus_ar_rain", f4r), ("F4_h1_corr_minus_ar_rain", f4c),
                 ("F5_h3_ar_rain_minus_persist", f5)):
        rows.append({"stat": s, "point_m": d[0], "lo_m": d[1], "hi_m": d[2]})
    pd.DataFrame(rows).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.5f")

    f = lambda d: f"{d[0] * 1000:+.2f} mm [{d[1] * 1000:+.2f}, {d[2] * 1000:+.2f}]"
    L = ["[강남 2026] T1 시공간 GNN 단기 예측 판정 (사전 고정 규칙: src/analysis/t1_st_rules.py, docs/17)", "=" * 78,
         f"범위: 서울시 강남구 센서 {len(nodes)}개(예측 대상) + 입력 센서, 2026-06-01~08-31, 5폴드. 평가 이벤트 {len(res[H]['ev_ids'])}개. "
         f"부트스트랩 {N_BOOT}회, 시드 {BOOT_SEED}. GNN 그래프 2개(도로거리, 상관)를 비교했다(다중 비교 보정 없음).",
         "그래프는 실제 관로 연결이 아니다. 상관 그래프는 센서가 있는 곳에서만 만들 수 있다.", "", "[판정]",
         f"  F1 30분 G-road − M2: {f(f1)} → {v1}",
         f"  F2 30분 G-corr − M2: {f(f2)} → {v2}",
         f"  F3 30분 G-road − M3: {f(f3r)} → {'그래프 구조 효과' if g3r else '그래프 구조 효과로 보고하지 않음'}",
         f"     30분 G-corr − M3: {f(f3c)} → {'그래프 구조 효과' if g3c else '그래프 구조 효과로 보고하지 않음'}",
         f"  F4 10분 G-road − M2: {f(f4r)} → {f_verdict(*f4r[1:])}",
         f"     10분 G-corr − M2: {f(f4c)} → {f_verdict(*f4c[1:])}",
         f"  F5 30분 M2 − M0: {f(f5)} → {f5_verdict(*f5[1:])}"]
    for h in HORIZONS:
        r = res[h]
        L += ["", f"[참고: {h * 10}분 뒤, 이벤트 MAE (mm), 95% CI | 전체 기간 MAE (mm)] 이벤트 유효 표본 {r['n']:,}"]
        for m in MODELS:
            lo, hi = ci(r["boot"][:, k[m]])
            L.append(f"  {LABEL[m]:<16} {r['point'][k[m]] * 1000:7.2f}  [{lo * 1000:.2f}, {hi * 1000:.2f}] | {r['all_mae'][m] * 1000:6.2f}")
    L += ["", f"[참고: 센서별 30분 뒤 이벤트 MAE (mm)]"]
    for j, n in enumerate(nodes):
        L.append(f"  {n}: " + ", ".join(f"{LABEL[m].split()[0]} {res[H]['mae_j'][j, k[m]] * 1000:.1f}" for m in MODELS))
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
