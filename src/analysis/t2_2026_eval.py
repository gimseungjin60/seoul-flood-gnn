"""2026 우기 재비교 판정: 사전 고정 규칙(src/analysis/t2_2026_rules.py, docs/11)의 P1·P2·P3·S1만 판정하고 나머지는 참고로 보고한다.
결과를 보기 전에 작성했다(2026-11-02). 판정 함수·상수는 t2_2026_rules 에서만 가져온다.

실행: python -m src.analysis.t2_2026_eval   (먼저 python -m src.analysis.t2_2026 --part loso / gnn 이 끝나야 한다. --part h 결과는 있으면 참고로 붙인다)
출력: data/processed/gangnam2026_t2_report.txt, gangnam2026_t2_ci.csv

유효 블록 = 정답이 있고 이웃 정보가 있는 블록(2025 t2_loso·t2_gnn_eval 과 같은 정의).
부트스트랩 = 평가 이벤트 복원추출 2000회(시드 0), 타깃 센서 평균. P3 은 같은 시드로 같은 이벤트 추출을 재현한다.
"""
import numpy as np
import pandas as pd

from src.analysis.t2_2026 import OUT_PRED
from src.analysis.t2_2026_rules import (BOOT_SEED, N_BOOT, PEAK_MIN_BLOCKS, S1_IMPROVE, p1_verdict, p2_graph_effect, p3_eligible,
                                       p3_verdict, s1_min_sensors, s1_verdict)
from src.analysis.t2_gnn_eval import NAMES, SHORT, error_sums
from src.analysis.t2_loso import event_blocks
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t2_report.txt"
OUT_CI = PROCESSED / "gangnam2026_t2_ci.csv"
RESULT_2025 = "2025(센서 18개, 이벤트 8개): GNN − IDW +0.80 mm [−0.29, +2.08] 구분 불가, GNN − (4) −0.19 mm [−0.53, +0.22]"


def load_preds(loso_path, gnn_path):
    z, g = np.load(loso_path, allow_pickle=True), np.load(gnn_path, allow_pickle=True)
    if not (np.array_equal(z["bt"], g["bt"]) and list(z["nodes"]) == list(g["nodes"])):
        raise ValueError("기준 모델 예측과 GNN 예측의 시각·센서 순서가 다르다")
    actual = z["actual_m"].astype(float)
    if not np.allclose(np.nan_to_num(actual), np.nan_to_num(g["actual_m"].astype(float)), atol=1e-4):
        raise ValueError("두 예측 파일의 정답이 다르다")
    pred = {m: z[f"pred_{m}"].astype(float) for m in NAMES if m != "gnn"}
    pred["gnn"] = g["pred_gnn"].astype(float)
    return pd.DatetimeIndex(z["bt"]), [str(n) for n in z["nodes"]], actual, pred


def boot_counts(E: int) -> np.ndarray:
    """t2_loso_bootstrap.bootstrap 과 같은 추출(같은 시드·같은 순서)."""
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, E, size=(N_BOOT, E))
    return np.stack([np.bincount(d, minlength=E) for d in draws]).astype(float)


def peak_arrays(pred, actual, bt, events, models):
    """(E, J, M) 첨두 오차(mm)·첨두 시각 오차(분), (E, J) 유효(블록 >= 6), (E, J) P3 대상(실제 첨두 상승 >= 50 mm)."""
    ok = np.isfinite(actual) & np.isfinite(pred["mean_others"])
    ev = event_blocks(bt, events)
    E, J = len(ev), actual.shape[1]
    pk, tk = np.full((E, J, len(models)), np.nan), np.full((E, J, len(models)), np.nan)
    valid, elig = np.zeros((E, J), bool), np.zeros((E, J), bool)
    for e, (_, idx) in enumerate(ev):
        for j in range(J):
            ii = idx[ok[idx, j]]
            if len(ii) < PEAK_MIN_BLOCKS:
                continue
            a = actual[ii, j]
            valid[e, j], elig[e, j] = True, p3_eligible(a)
            for k, m in enumerate(models):
                p = pred[m][ii, j]
                pk[e, j, k] = abs(np.nanmax(p) - np.nanmax(a)) * 1000
                tk[e, j, k] = abs(int(np.nanargmax(p)) - int(np.nanargmax(a))) * 10
    return pk, tk, valid, elig


def p3_stat(tk_m: np.ndarray, elig: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """counts (R, E) → (R,) 센서별 대상 조합 시각 오차 평균의 센서 평균(대상 조합이 있는 센서만)."""
    x = np.where(elig, tk_m, 0.0)
    num = np.einsum("re,ej->rj", counts, x)
    den = np.einsum("re,ej->rj", counts, elig.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(den > 0, num / den, np.nan), axis=1)


def h_reference(events) -> list[str]:
    if not OUT_PRED["h"].exists():
        return ["  H 실험 결과 없음(python -m src.analysis.t2_2026 --part h)"]
    h = np.load(OUT_PRED["h"], allow_pickle=True)
    bt, nodes, actual = pd.DatetimeIndex(h["bt"]), [str(n) for n in h["nodes"]], h["actual_m"].astype(float)
    pred = {m: h[f"pred_{m}"].astype(float) for m in NAMES}
    S, N, _ = error_sums(pred, actual, bt, events)
    point, mae_j = macro_mae(S, N, np.ones((1, S.shape[0])))
    boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)
    k = {m: i for i, m in enumerate(NAMES)}
    lo, hi = ci(boot[:, k["gnn"]] - boot[:, k["idw_p1"]])
    L = [f"  센서 {len(nodes)}개 {nodes}, H를 학습(가짜 타깃·이웃 입력·표준화 통계)에서 완전히 뺀 모델. 참고(판정 아님)",
         "  이벤트 MAE(mm): " + ", ".join(f"{SHORT[m]} {point[0, k[m]] * 1000:.2f}" for m in NAMES),
         f"  GNN − IDW {(point[0, k['gnn']] - point[0, k['idw_p1']]) * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]"]
    for j, n in enumerate(nodes):
        L.append(f"    {n}: " + ", ".join(f"{SHORT[m]} {mae_j[0, j, k[m]] * 1000:.2f}" for m in ("idw_p1", "ridge_rain_time_nbr", "gnn")))
    return L


def main() -> None:
    bt, nodes, actual, pred = load_preds(OUT_PRED["loso"], OUT_PRED["gnn"])
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    S, N, ev_ids = error_sums(pred, actual, bt, events)
    E, J = S.shape[0], S.shape[1]
    k = {m: i for i, m in enumerate(NAMES)}
    point, mae_j = macro_mae(S, N, np.ones((1, E)))
    point, mae_j = point[0], mae_j[0]
    boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)

    def diff(a, b):
        lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
        return float(point[k[a]] - point[k[b]]), lo, hi

    d1, d2 = diff("gnn", "idw_p1"), diff("gnn", "ridge_rain_time_nbr")
    v1 = p1_verdict(d1[1], d1[2])
    v2 = p2_graph_effect(v1, d2[2])

    pk_models = ["idw_p1", "ridge_rain_time_nbr", "hgb_rain_time_nbr", "gnn"]
    pk, tk, valid, elig = peak_arrays(pred, actual, bt, events, pk_models)
    cnt = boot_counts(E)
    gi, ii = pk_models.index("gnn"), pk_models.index("idw_p1")
    p3_point = float(p3_stat(tk[:, :, gi], elig, np.ones((1, E)))[0] - p3_stat(tk[:, :, ii], elig, np.ones((1, E)))[0])
    p3_lo, p3_hi = ci(p3_stat(tk[:, :, gi], elig, cnt) - p3_stat(tk[:, :, ii], elig, cnt))
    v3 = p3_verdict(p3_lo, p3_hi)

    g4 = 1 - mae_j[:, k["ridge_rain_time_nbr"]] / mae_j[:, k["ridge_rain_time"]]
    g5 = 1 - mae_j[:, k["hgb_rain_time_nbr"]] / mae_j[:, k["ridge_rain_time"]]
    n_imp = int(((g4 >= S1_IMPROVE) | (g5 >= S1_IMPROVE)).sum())
    v4 = s1_verdict(n_imp, J)

    rows = [{"stat": f"mae_{m}", "point_m": point[k[m]], "lo_m": ci(boot[:, k[m]])[0], "hi_m": ci(boot[:, k[m]])[1]} for m in NAMES]
    rows += [{"stat": "P1_gnn_minus_idw", "point_m": d1[0], "lo_m": d1[1], "hi_m": d1[2]},
             {"stat": "P2_gnn_minus_ridge_nbr", "point_m": d2[0], "lo_m": d2[1], "hi_m": d2[2]},
             {"stat": "P3_peak_time_gnn_minus_idw_min", "point_m": p3_point, "lo_m": p3_lo, "hi_m": p3_hi}]
    pd.DataFrame(rows).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.5f")

    L = []
    w = L.append
    w("[강남 2026] T2 재비교 판정 (사전 고정 규칙: src/analysis/t2_2026_rules.py, docs/11)")
    w("=" * 78)
    w(f"범위: 서울시 강남구 센서 {J}개(타깃) + 이웃 후보, 2026-06-01~08-31. 평가 이벤트 {E}개 {ev_ids}. "
      f"부트스트랩 {N_BOOT}회, 시드 {BOOT_SEED}, 타깃 평균.")
    w("이 검증은 타깃의 기저선·규모를 안다는 가정 하의 상대 상승폭 추정이며 절대수위 추정이 아니다. 그래프는 도로거리 근사(실제 관로 아님).")
    w("\n[판정]")
    w(f"  P1 GNN − IDW 이벤트 MAE: {d1[0] * 1000:+.2f} mm [{d1[1] * 1000:+.2f}, {d1[2] * 1000:+.2f}] → {v1}")
    w(f"  P2 GNN − (4) 강우+이웃(선형): {d2[0] * 1000:+.2f} mm [{d2[1] * 1000:+.2f}, {d2[2] * 1000:+.2f}] → "
      f"{'그래프 구조 효과' if v2 else '그래프 구조 효과로 보고하지 않음'}")
    w(f"  P3 GNN − IDW 첨두 시각 오차: {p3_point:+.1f}분 [{p3_lo:+.1f}, {p3_hi:+.1f}] → {v3} "
      f"(대상 조합 {int(elig.sum())}/{int(valid.sum())}, 실제 첨두 상승 >= 50 mm)")
    w(f"  S1 (4) 또는 (5)가 (3) 대비 이벤트 MAE {S1_IMPROVE:.0%} 이상 감소한 타깃: {n_imp}/{J} (기준 {s1_min_sensors(J)}) → "
      f"{'T2 공간 신호 있음' if v4 else 'T2 공간 신호 기준 미달'}")
    w("\n[참고: 모델별 이벤트 MAE (mm), 95% CI]")
    for m in NAMES:
        lo, hi = ci(boot[:, k[m]])
        w(f"  {SHORT[m]:<16} {point[k[m]] * 1000:7.2f}  [{lo * 1000:.2f}, {hi * 1000:.2f}]")
    w(f"\n[참고: 2025와의 비교, 서술용] {RESULT_2025}")
    w("\n[참고: 첨두 지표, 이벤트·센서 평균(유효 조합 전체), 판정 아님]")
    for c, m in enumerate(pk_models):
        w(f"  {SHORT[m]:<16} 첨두 오차 {np.nanmean(np.where(valid, pk[:, :, c], np.nan)):6.1f} mm, "
          f"첨두 시각 오차 {np.nanmean(np.where(valid, tk[:, :, c], np.nan)):6.1f}분")
    w("\n[참고: 센서별 이벤트 MAE (mm)]")
    for j, n in enumerate(nodes):
        w(f"  {n}: " + ", ".join(f"{SHORT[m]} {mae_j[j, k[m]] * 1000:.2f}" for m in ("idw_p1", "ridge_rain_time", "ridge_rain_time_nbr",
                                                                                   "hgb_rain_time_nbr", "gnn"))
          + f" | (4)/(5) 개선 {g4[j]:+.0%}/{g5[j]:+.0%}")
    w("\n[참고: 신규 센서 검증 H]")
    L.extend(h_reference(events))
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
