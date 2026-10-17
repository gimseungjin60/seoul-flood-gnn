"""T2 LOSO 결과의 이벤트 단위 부트스트랩: 모델별 이벤트 MAE 95% CI와 IDW 등과의 차이 CI (GNN 아님, test 미사용).

실행: python -m src.analysis.t2_loso_bootstrap   (먼저 python -m src.analysis.t2_loso 로 예측 저장)
입력: data/processed/gangnam_t2_loso_predictions.npz, gangnam_rain_events.csv
출력: data/processed/gangnam_t2_loso_bootstrap_{ci.csv,report.txt}, outputs/figures/fig9_t2_loso_event_mae_ci.png

규칙 (결과를 보기 전에 정함)
  표본 단위 = 평가 이벤트(train+val 의 8개). 이벤트를 복원추출 2000회(시드 0)한다. 한 반복에서 센서별 이벤트 MAE는
  뽑힌 이벤트들의 블록 오차를 합쳐 계산(기존 '풀링 이벤트 MAE'와 같은 정의)하고, 모델 통계는 18개 센서 평균(매크로 평균)이다.
  CI = 부트스트랩 분포의 2.5/97.5 백분위. 두 모델 차이(A-B)의 CI 가 0을 포함하면 '통계적으로 구분 불가'로 보고한다.
  비교: (4)-(2), (5)-(2) [IDW 대비], (4)-(1), (5)-(1), (4)-(3), (5)-(3), (2)-(3), (1)-(3), (5)-(3b).
  보조(판정 아님): 사전 기준 '(4) 또는 (5)가 (3) 대비 10% 이상 줄이는 센서 수'의 부트스트랩 분포와 P(수 >= 6),
                  이벤트 하나씩 빼본 민감도(leave-one-event-out).
한계: 이벤트가 8개뿐이라 부트스트랩 분포가 이산적이고 CI 가 넓다. 큰 사건(07-16~17, 154.8 mm) 한 개가 오차의 큰 비중을 차지할 수 있다.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.t2_loso import IMPROVE, MIN_SENSORS, MODELS, OUT_PRED, event_blocks  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.prepare_rainfall import OUT_EVENTS  # noqa: E402

N_BOOT, SEED = 2000, 0
OUT_CI = PROCESSED / "gangnam_t2_loso_bootstrap_ci.csv"
REPORT = PROCESSED / "gangnam_t2_loso_bootstrap_report.txt"
SHORT = {"mean_others": "(1) 전체 평균", "idw_p1": "(2) IDW", "ridge_rain_time": "(3) 강우+시간대",
         "ridge_rain_time_nbr": "(4) +이웃 Ridge", "hgb_rain_time_nbr": "(5) +이웃 HGB", "hgb_rain_time": "(3b) 강우+시간대 HGB"}
DIFFS = [("ridge_rain_time_nbr", "idw_p1"), ("hgb_rain_time_nbr", "idw_p1"), ("ridge_rain_time_nbr", "mean_others"),
         ("hgb_rain_time_nbr", "mean_others"), ("ridge_rain_time_nbr", "ridge_rain_time"), ("hgb_rain_time_nbr", "ridge_rain_time"),
         ("idw_p1", "ridge_rain_time"), ("mean_others", "ridge_rain_time"), ("hgb_rain_time_nbr", "hgb_rain_time")]


def event_error_sums(pred: dict, actual: np.ndarray, bt: pd.DatetimeIndex, events: pd.DataFrame):
    """이벤트별·센서별·모델별 절대오차 합 S (E, J, M)와 블록 수 N (E, J). 이벤트 창끼리 겹치면 이중 집계되므로 에러."""
    ev = event_blocks(bt, events)
    seen = np.zeros(len(bt), dtype=int)
    for _, idx in ev:
        seen[idx] += 1
    if (seen > 1).any():
        raise ValueError("이벤트 창이 서로 겹친다: 복원추출에서 블록이 이중 집계된다")
    J = actual.shape[1]
    ok = np.isfinite(actual) & np.isfinite(pred["mean_others"])                  # summarize()와 같은 유효 블록 정의
    S = np.zeros((len(ev), J, len(MODELS)))
    N = np.zeros((len(ev), J))
    for e, (_, idx) in enumerate(ev):
        v = ok[idx]
        N[e] = v.sum(axis=0)
        for k, m in enumerate(MODELS):
            err = np.abs(pred[m][idx] - actual[idx])
            S[e, :, k] = np.where(v, err, 0.0).sum(axis=0)
    return S, N, [e for e, _ in ev]


def macro_mae(S: np.ndarray, N: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """counts (B, E) 이벤트 선택 횟수 → (B, M) 센서 평균 이벤트 MAE, (B, J, M) 센서별 MAE."""
    s = np.einsum("be,ejm->bjm", counts, S)
    n = np.einsum("be,ej->bj", counts, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = s / n[:, :, None]
    return np.nanmean(mae, axis=1), mae


def bootstrap(S, N, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    E = S.shape[0]
    draws = rng.integers(0, E, size=(n_boot, E))
    counts = np.stack([np.bincount(d, minlength=E) for d in draws]).astype(float)
    return macro_mae(S, N, counts)


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def pass_count(mae_bjm: np.ndarray) -> np.ndarray:
    """부트스트랩 반복마다 사전 기준 통과 센서 수. mae (B, J, M)."""
    k = {m: i for i, m in enumerate(MODELS)}
    g4 = 1 - mae_bjm[:, :, k["ridge_rain_time_nbr"]] / mae_bjm[:, :, k["ridge_rain_time"]]
    g5 = 1 - mae_bjm[:, :, k["hgb_rain_time_nbr"]] / mae_bjm[:, :, k["ridge_rain_time"]]
    return ((g4 >= IMPROVE) | (g5 >= IMPROVE)).sum(axis=1)


def make_figure(point, cis, diffs):
    from src.visualization.make_figures import C, _note, _save
    order = ["mean_others", "idw_p1", "ridge_rain_time", "ridge_rain_time_nbr", "hgb_rain_time_nbr"]
    k = {m: i for i, m in enumerate(MODELS)}
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(order))
    vals = [point[k[m]] * 1000 for m in order]
    lo = [point[k[m]] * 1000 - cis[m][0] * 1000 for m in order]
    hi = [cis[m][1] * 1000 - point[k[m]] * 1000 for m in order]
    ax.bar(x, vals, width=0.58, color=C["s1"], zorder=3)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=C["ink"], elinewidth=1.6, capsize=5, zorder=4)
    for xi, m, v, h in zip(x, order, vals, hi):
        ax.text(xi, v + h + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=11, color=C["ink"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m].replace(" ", "\n", 1) for m in order])
    ax.set_ylabel("이벤트 구간 MAE (mm), 센서 18개 평균 (낮을수록 좋음)")
    ax.set_ylim(0, max(np.array(vals) + np.array(hi)) * 1.22)
    ax.grid(axis="x", visible=False)
    d4, d5 = diffs[("ridge_rain_time_nbr", "idw_p1")], diffs[("hgb_rain_time_nbr", "idw_p1")]
    both_zero = all(d[0] <= 0 <= d[1] for d in (d4, d5))
    both_worse = all(d[0] > 0 for d in (d4, d5))
    if both_zero:
        t = "이웃 센서를 쓰면 강우+시간대 모델보다 오차가 줄지만, 학습 모델은 단순 IDW와 통계적으로 구분되지 않는다"
    elif both_worse:
        t = "이웃 센서를 쓰면 강우+시간대 모델보다 오차가 줄지만, 학습 모델은 단순 IDW보다 약간 나쁘다"
    else:
        t = "이웃 센서를 쓰면 강우+시간대 모델보다 오차가 줄지만, IDW와의 우열은 모델마다 다르다"
    ax.set_title(t, fontsize=12.5)
    _note(fig, f"막대=점추정, 선=95% CI (평가 이벤트 8개를 2000회 복원추출). IDW 대비 차이 95% CI: (4) {d4[0] * 1000:+.1f}~{d4[1] * 1000:+.1f} mm, "
               f"(5) {d5[0] * 1000:+.1f}~{d5[1] * 1000:+.1f} mm.\n"
               "서울시 강남구 센서 18개, 센서 1개를 통째로 제외(LOSO), 타깃 과거값 미사용. 이력이 있던 센서의 기저선·규모를 안다는 가정 하의 상대 상승폭 추정이며 절대수위 추정이 아니다.", y=-0.05)
    return _save(fig, "fig9_t2_loso_event_mae_ci.png")


def main() -> None:
    t0 = time.time()
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    pred = {m: z[f"pred_{m}"].astype(float) for m in MODELS}
    actual = z["actual_m"].astype(float)
    nodes = [str(x) for x in z["nodes"]]
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    S, N, ev_ids = event_error_sums(pred, actual, bt, events)
    E = S.shape[0]
    k = {m: i for i, m in enumerate(MODELS)}

    point_macro, point_sensor = macro_mae(S, N, np.ones((1, E)))
    point = point_macro[0]
    boot_macro, boot_sensor = bootstrap(S, N)
    cis = {m: ci(boot_macro[:, k[m]]) for m in MODELS}
    diffs = {}
    for a, b in DIFFS:
        d = boot_macro[:, k[a]] - boot_macro[:, k[b]]
        diffs[(a, b)] = ci(d)
    pc_point = int(pass_count(point_sensor)[0])
    pc = pass_count(boot_sensor)
    loo = []
    for drop in range(E):
        c = np.ones((1, E))
        c[0, drop] = 0
        loo.append((ev_ids[drop], int(pass_count(macro_mae(S, N, c)[1])[0]), macro_mae(S, N, c)[0][0]))

    rows = [{"model": m, "label": SHORT[m], "macro_event_mae_m": point[k[m]], "ci_lo": cis[m][0], "ci_hi": cis[m][1]} for m in MODELS]
    pd.DataFrame(rows).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.6f")

    make_figure(point, cis, diffs)

    L = []
    w = L.append
    w("[강남] T2 LOSO 이벤트 부트스트랩 (GNN 아님, test 구간 미사용)")
    w("=" * 74)
    w("※ 규칙은 src/analysis/t2_loso_bootstrap.py 상단 docstring 참조(결과 확인 전 작성).")
    w(f"평가 이벤트 {E}개 {ev_ids}를 복원추출 {N_BOOT}회(시드 {SEED}). 통계 = 센서 18개 평균의 이벤트 구간 MAE(m).")
    w("")
    w("1. 모델별 이벤트 MAE (점추정, 95% CI)")
    for m in MODELS:
        w(f"  {SHORT[m]:22s} {point[k[m]]:.4f} m  [{cis[m][0]:.4f}, {cis[m][1]:.4f}]")
    w("")
    w("2. 모델 간 차이 (A - B, 음수면 A가 더 낮은(좋은) 오차) 와 95% CI")
    w(f"  {'A - B':36s} {'점추정':>9s} {'95% CI':>22s}   판정")
    for a, b in DIFFS:
        d = point[k[a]] - point[k[b]]
        lo, hi = diffs[(a, b)]
        tag = "통계적으로 구분 불가 (CI가 0 포함)" if lo <= 0 <= hi else ("A가 낮음(좋음)" if hi < 0 else "A가 높음(나쁨)")
        w(f"  {SHORT[a] + ' - ' + SHORT[b]:36s} {d:+9.4f} [{lo:+8.4f}, {hi:+8.4f}]   {tag}")
    w("")
    w("3. 이벤트별 센서 평균 MAE (m)와 큰 사건의 비중")
    tot = S.sum(axis=(0, 1))
    w(f"  {'event':>5s} {'블록수(센서합)':>14s} " + " ".join(f"{SHORT[m][:5]:>8s}" for m in MODELS) + "   IDW 총오차 비중")
    for e, eid in enumerate(ev_ids):
        mm = [np.nanmean(S[e, :, k[m]] / np.where(N[e] > 0, N[e], np.nan)) for m in MODELS]
        w(f"  {eid:5d} {int(N[e].sum()):14d} " + " ".join(f"{v:8.4f}" for v in mm) + f"   {S[e, :, k['idw_p1']].sum() / tot[k['idw_p1']]:.1%}")
    w("")
    w("4. 보조: 사전 기준 '(4) 또는 (5)가 (3) 대비 10% 이상 줄인 센서 수'의 안정성 (판정에는 사용하지 않음)")
    w(f"  점추정 {pc_point}개, 부트스트랩 중앙값 {int(np.median(pc))}개, 95% 구간 [{int(np.percentile(pc, 2.5))}, {int(np.percentile(pc, 97.5))}], "
      f"P(수 >= {MIN_SENSORS}) = {float((pc >= MIN_SENSORS).mean()):.1%}")
    w("  이벤트 하나씩 제외(leave-one-event-out): " + ", ".join(f"event {e} 제외 → {n}개" for e, n, _ in loo))
    w("")
    w("주의")
    w("  - 이벤트 8개를 복원추출하므로 한 이벤트가 여러 번 뽑히거나 빠진다. 큰 사건 하나(예: event 5)가 빠진 반복에서는 결과가 크게 달라질 수 있다.")
    w("  - CI는 이벤트(사건) 단위 불확실성만 반영한다. 센서·시간 블록의 비독립, 5폴드 교차검증의 학습 구성 차이, 라벨 기저선 가정은 반영하지 않는다.")
    w("  - '구분 불가'는 '같다'가 아니라 이 표본으로는 차이를 확인할 수 없다는 뜻이다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
