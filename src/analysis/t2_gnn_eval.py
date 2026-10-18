"""T2 GNN 판정: 사전 고정 규칙(src/analysis/t2_gnn.py 상단 docstring)대로 GNN − IDW, GNN − (4) 차이의 95% CI 를 계산한다.

실행: python -m src.analysis.t2_gnn_eval   (먼저 python -m src.analysis.t2_loso 와 python -m src.analysis.t2_gnn 이 끝나 있어야 한다)
입력: data/processed/gangnam_t2_loso_predictions.npz (기존 모델 예측), gangnam_t2_gnn_predictions.npz, gangnam_rain_events.csv
출력: data/processed/gangnam_t2_gnn_report.txt, gangnam_t2_gnn_ci.csv

방법(기존 t2_loso_bootstrap 과 동일): 평가 이벤트(train+val 8개)를 2000회 복원추출(시드 0), 센서 18개 평균 이벤트 MAE.
유효 블록 = 정답이 있고 이웃 정보가 있는 블록(기존 정의). test 구간은 읽지 않는다.
판정은 t2_gnn.py 에 사전 고정한 두 기준뿐이다. 나머지(val 이벤트, 큰 사건 제외, 이벤트 하나씩 빼기, 센서별 비교)는 참고이며 판정에 쓰지 않는다.
"""
import time

import numpy as np
import pandas as pd

from src.analysis.t2_gnn import BOOT_SEED, N_BOOT, OUT_CI, OUT_PRED as GNN_PRED, OUT_REPORT
from src.analysis.t2_loso import OUT_PRED as LOSO_PRED, event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_EVENTS

NAMES = ["mean_others", "idw_p1", "ridge_rain_time", "ridge_rain_time_nbr", "hgb_rain_time_nbr", "gnn"]
SHORT = {"mean_others": "(1) 전체 평균", "idw_p1": "(2) IDW", "ridge_rain_time": "(3) 강우+시간대", "ridge_rain_time_nbr": "(4) 강우+이웃(선형)",
         "hgb_rain_time_nbr": "(5) 강우+이웃(트리)", "gnn": "GNN"}


def primary_verdict(lo: float, hi: float) -> str:
    """주 판정 (사전 고정): GNN − IDW 차이의 95% CI. 음수가 GNN이 낮은(좋은) 쪽."""
    if hi < 0:
        return "GNN이 IDW를 넘음"
    if lo > 0:
        return "IDW보다 나쁨"
    return "구분 불가"


def secondary_ok(primary: str, hi_vs4: float) -> bool:
    """보조 판정 (사전 고정): 주 판정이 '넘음'이고 GNN − (4) 의 CI 상한도 0보다 작아야 '그래프 구조 효과'."""
    return primary == "GNN이 IDW를 넘음" and hi_vs4 < 0


def error_sums(pred: dict, actual: np.ndarray, bt: pd.DatetimeIndex, events: pd.DataFrame):
    ev = event_blocks(bt, events)
    seen = np.zeros(len(bt), dtype=int)
    for _, idx in ev:
        seen[idx] += 1
    if (seen > 1).any():
        raise ValueError("이벤트 창이 서로 겹친다")
    ok = np.isfinite(actual) & np.isfinite(pred["mean_others"])
    for m in NAMES:                                                              # 모든 모델이 같은 유효 블록에서 값이 있어야 한다
        bad = ok & ~np.isfinite(pred[m])
        if bad.any():
            raise ValueError(f"{m}: 유효 블록 {int(bad.sum())}개에 예측이 없다")
    J = actual.shape[1]
    S = np.zeros((len(ev), J, len(NAMES)))
    N = np.zeros((len(ev), J))
    for e, (_, idx) in enumerate(ev):
        v = ok[idx]
        N[e] = v.sum(axis=0)
        for k, m in enumerate(NAMES):
            S[e, :, k] = np.where(v, np.abs(pred[m][idx] - actual[idx]), 0.0).sum(axis=0)
    return S, N, [e for e, _ in ev]


def main() -> None:
    t0 = time.time()
    z = np.load(LOSO_PRED, allow_pickle=True)
    g = np.load(GNN_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    if not (np.array_equal(bt.to_numpy(), pd.DatetimeIndex(g["bt"]).to_numpy()) and list(z["nodes"]) == list(g["nodes"])):
        raise ValueError("기존 LOSO 예측과 GNN 예측의 시각·센서 순서가 다르다")
    actual = z["actual_m"].astype(float)
    if not np.allclose(np.nan_to_num(actual), np.nan_to_num(g["actual_m"].astype(float)), atol=1e-4):
        raise ValueError("두 예측 파일의 정답이 다르다")
    pred = {m: z[f"pred_{m}"].astype(float) for m in NAMES if m != "gnn"}
    pred["gnn"] = g["pred_gnn"].astype(float)
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    S, N, ev_ids = error_sums(pred, actual, bt, events)
    k = {m: i for i, m in enumerate(NAMES)}
    split = dict(zip(events.event_id, events.split_by_start))
    E = len(ev_ids)

    point, mae_j = macro_mae(S, N, np.ones((1, E)))
    point = point[0]
    boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)

    def diff_ci(a, b):
        d = boot[:, k[a]] - boot[:, k[b]]
        lo, hi = ci(d)
        return float(point[k[a]] - point[k[b]]), lo, hi

    d_idw = diff_ci("gnn", "idw_p1")
    d_4 = diff_ci("gnn", "ridge_rain_time_nbr")
    verdict = primary_verdict(d_idw[1], d_idw[2])
    graph_effect = secondary_ok(verdict, d_4[2])

    L = []
    w = L.append
    w("[강남] T2 GNN 판정 (사전 고정 규칙은 src/analysis/t2_gnn.py 상단 docstring, test 구간 미사용)")
    w("=" * 78)
    w(f"평가 이벤트 {E}개: {ev_ids} (분할: {[split[e] for e in ev_ids]}). 부트스트랩 {N_BOOT}회, 시드 {BOOT_SEED}. 센서 18개 평균 이벤트 MAE(mm).")
    w("범위: 서울시 강남구 센서 18개(2025-06~08). 이 검증은 타깃의 기저선·규모를 안다는 가정 하의 상대 상승폭 추정이며 절대수위 추정이 아니다.")
    w("")
    w("[이벤트 MAE (mm), 점추정과 95% CI]")
    rows = []
    for m in NAMES:
        lo, hi = ci(boot[:, k[m]])
        w(f"  {SHORT[m]:<18} {point[k[m]] * 1000:6.2f}  [{lo * 1000:6.2f}, {hi * 1000:6.2f}]")
        rows.append({"model": m, "point_mm": point[k[m]] * 1000, "ci_lo_mm": lo * 1000, "ci_hi_mm": hi * 1000})
    w("")
    w("[판정 — 사전 고정 기준]")
    w(f"  주 판정  GNN − IDW      = {d_idw[0] * 1000:+.2f} mm  95% CI [{d_idw[1] * 1000:+.2f}, {d_idw[2] * 1000:+.2f}]  → {verdict}")
    w(f"  보조 판정 GNN − (4)선형  = {d_4[0] * 1000:+.2f} mm  95% CI [{d_4[1] * 1000:+.2f}, {d_4[2] * 1000:+.2f}]")
    if graph_effect:
        w("  결론: 그래프 구조 효과 인정 (주 판정 '넘음' + 보조 CI 상한 < 0)")
    elif verdict == "GNN이 IDW를 넘음":
        w("  결론: IDW는 넘었으나 선형 이웃 모델(4)과 구분 불가 → '그래프 구조 효과'로는 보고하지 않는다")
    elif verdict == "구분 불가":
        w("  결론: GNN은 IDW와 통계적으로 구분되지 않는다 → 그래프 구조 효과로 보고하지 않는다")
    else:
        w("  결론: GNN이 IDW보다 나쁘다 → 그래프 구조 효과로 보고하지 않는다")
    rows.append({"model": "diff_gnn_minus_idw", "point_mm": d_idw[0] * 1000, "ci_lo_mm": d_idw[1] * 1000, "ci_hi_mm": d_idw[2] * 1000, "verdict": verdict})
    rows.append({"model": "diff_gnn_minus_ridge_nbr4", "point_mm": d_4[0] * 1000, "ci_lo_mm": d_4[1] * 1000, "ci_hi_mm": d_4[2] * 1000,
                 "verdict": "그래프 구조 효과 인정" if graph_effect else "인정하지 않음"})
    pd.DataFrame(rows).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.4f")

    w("")
    w("[참고 — 판정에 쓰지 않음]")
    val_ev = [e for e in ev_ids if split[e] == "val"]
    w(f"  val 이벤트 {val_ev}(하이퍼파라미터 선택에 쓰이지 않음, 이벤트 2개뿐이라 CI를 계산하지 않고 이벤트별 차이만 본다): GNN − IDW / GNN − (4) (mm, 센서 평균)")
    for e in val_ev:
        sel = np.zeros((1, E))
        sel[0, ev_ids.index(e)] = 1
        p, _ = macro_mae(S, N, sel)
        w(f"    이벤트 {e}: GNN − IDW {1000 * (p[0, k['gnn']] - p[0, k['idw_p1']]):+.2f}, GNN − (4) {1000 * (p[0, k['gnn']] - p[0, k['ridge_rain_time_nbr']]):+.2f}")
    sel = np.zeros((1, E))
    sel[0, [ev_ids.index(e) for e in val_ev]] = 1
    p, _ = macro_mae(S, N, sel)
    w(f"    val 이벤트 합친 값: GNN − IDW {1000 * (p[0, k['gnn']] - p[0, k['idw_p1']]):+.2f}, GNN − (4) {1000 * (p[0, k['gnn']] - p[0, k['ridge_rain_time_nbr']]):+.2f}")
    big = [e for e in (5, 8) if e in ev_ids]
    keep = np.array([0.0 if e in big else 1.0 for e in ev_ids])[None]
    p, _ = macro_mae(S, N, keep)
    w(f"  큰 사건 {big} 제외({int(keep.sum())}개): GNN − IDW {1000 * (p[0, k['gnn']] - p[0, k['idw_p1']]):+.2f}, GNN − (4) {1000 * (p[0, k['gnn']] - p[0, k['ridge_rain_time_nbr']]):+.2f} mm")
    ev6 = [i for i, e in enumerate(ev_ids) if e not in big]
    boot6, _ = bootstrap(S[ev6], N[ev6], N_BOOT, BOOT_SEED)
    for a_, lab in (("gnn", "GNN"), ("ridge_rain_time_nbr", "(4) 선형"), ("hgb_rain_time_nbr", "(5) 트리")):
        d6 = boot6[:, k[a_]] - boot6[:, k["idw_p1"]]
        lo6, hi6 = ci(d6)
        pt6 = macro_mae(S[ev6], N[ev6], np.ones((1, len(ev6))))[0][0]
        w(f"  큰 사건 {big} 제외 {lab} − IDW: {1000 * (pt6[k[a_]] - pt6[k['idw_p1']]):+.2f} mm, 95% CI [{lo6 * 1000:+.2f}, {hi6 * 1000:+.2f}] (이벤트 {len(ev6)}개 복원추출, 참고)")
    w("  이벤트 하나씩 뺐을 때 GNN − IDW (mm): " + ", ".join(
        f"-{e}:{1000 * (macro_mae(S, N, (1 - np.eye(E)[[ev_ids.index(e)]]))[0][0, k['gnn']] - macro_mae(S, N, (1 - np.eye(E)[[ev_ids.index(e)]]))[0][0, k['idw_p1']]):+.2f}" for e in ev_ids))
    sensors = list(z["nodes"])
    per = pd.DataFrame(mae_j[0] * 1000, index=sensors, columns=NAMES)
    w(f"  센서별로 GNN 이 IDW 보다 이벤트 MAE 가 낮은 센서: {int((per['gnn'] < per['idw_p1']).sum())}/18, (4) 보다 낮은 센서: {int((per['gnn'] < per['ridge_rain_time_nbr']).sum())}/18")
    w("")
    w("한계: 이벤트 8개뿐이라 CI가 넓다. 학습 풀이 17개 센서뿐이다. 하이퍼파라미터는 사전에 고정했고(그리드 없음) 사후 민감도 분석은 판정 이후에만 별도로 수행하며 판정에 쓰지 않는다.")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
