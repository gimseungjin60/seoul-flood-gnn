"""기준 모델 2종 (GNN 비교용, 그래프·이웃 정보 미사용).

실행: python -m src.models.baselines
A. persistence: 입력 창 [t-59, t] 안의 마지막 '원본 관측값'을 t+h 예측값으로 사용.
   창 안에 관측이 하나도 없으면 예측하지 않음(NaN) — 임의값으로 대체하지 않음.
B. node_ridge_ar: 노드별 선형 회귀(릿지). 특징 = 자기 노드 과거 60분 정규화 값(결측 0) + 관측 마스크 60개 + 절편.
   학습=train 분할, λ 선택=val MAE, 테스트=test. 정규화는 학습 구간 통계만 사용.

공통 평가쌍 (두 모델 동일): 분할 샘플 t × 노드 중 (정답 t+h가 원본 관측) AND (입력 창에 그 노드 관측 ≥1).
출력: data/processed/gangnam_baseline_metrics_overall.csv, gangnam_baseline_metrics_by_node.csv,
      gangnam_baseline_test_predictions_h{10,30}.npz, gangnam_baseline_report.txt
"""
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from src.data.build_forecast_dataset import split_bounds
from src.data.clean_nodes import PROCESSED
from src.data.forecast_inputs import (HORIZONS, W, load_base, load_samples, make_targets, metrics,
                                      normalize, verify_samples)

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)  # 1차 실행에서 100(격자 끝) 다수 선택 → 확장
HIGH_Q = (0.95, 0.99)  # 사전 고정: 노드별 학습 구간 관측값 분위수 이상인 정답 = 고수위 평가 부분집합
OUT_OVERALL = PROCESSED / "gangnam_baseline_metrics_overall.csv"
OUT_NODE = PROCESSED / "gangnam_baseline_metrics_by_node.csv"
OUT_PRED = {h: PROCESSED / f"gangnam_baseline_test_predictions_h{h}.npz" for h in HORIZONS}
REPORT = PROCESSED / "gangnam_baseline_report.txt"


def last_observed(values: np.ndarray, mask: np.ndarray, t_idx: np.ndarray, w: int = W) -> np.ndarray:
    """각 (t, 노드)에서 창 [t-w+1, t] 안 마지막 원본 관측값. 없으면 NaN."""
    T, N = values.shape
    idx = np.where(mask, np.arange(T)[:, None], -1)
    last_idx = np.maximum.accumulate(idx, axis=0)[t_idx]           # t 이하 마지막 관측 인덱스
    ok = (last_idx >= 0) & (t_idx[:, None] - last_idx < w)
    out = np.full((len(t_idx), N), np.nan)
    out[ok] = values[last_idx[ok], np.nonzero(ok)[1]]
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """절편(마지막 열)은 벌점 제외."""
    P = np.eye(X.shape[1]) * lam
    P[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def node_features(zn: np.ndarray, mask: np.ndarray, t: np.ndarray, j: int, w: int = W) -> np.ndarray:
    """노드 j 자기 과거 w분만: [정규화값(결측 0) w개, 관측마스크 w개, 1]. 다른 노드 미사용."""
    xv = sliding_window_view(zn[:, j], w)[t - w + 1]
    xo = sliding_window_view(mask[:, j].astype(np.float64), w)[t - w + 1]
    return np.hstack([xv, xo, np.ones((len(t), 1))])


def eval_mask(y_mask: np.ndarray, persist: np.ndarray) -> np.ndarray:
    return y_mask & np.isfinite(persist)


def run_horizon(h, values, mask, nodes, mean, std, bounds):
    s = load_samples(h)
    std_safe = np.where(std > 0, std, 1.0)
    parts = {name: s.loc[s["split"] == name, "t_idx"].to_numpy() for name in ("train", "val", "test")}
    zn = np.where(mask, normalize(values, mean, std), 0.0)  # make_inputs와 같은 규칙(결측 0 + 마스크)
    data = {}
    for name, t in parts.items():
        y, ym = make_targets(values, mask, t, h)
        persist = last_observed(values, mask, t)
        data[name] = dict(t=t, y=y, ym=ym, persist=persist, E=eval_mask(ym, persist))

    # B: 노드별 릿지 — train으로 적합, val MAE로 λ 선택, test 평가
    ridge_pred = {k: np.full(data[k]["y"].shape, np.nan) for k in ("val", "test")}
    chosen = {}
    for j in range(len(nodes)):
        tr = data["train"]
        rows = tr["ym"][:, j]
        Xtr = node_features(zn, mask, tr["t"], j)[rows]
        ytr = normalize(tr["y"][rows, j], mean[j], std[j])
        Xva = node_features(zn, mask, data["val"]["t"], j)
        Ev = data["val"]["E"][:, j]
        best = None
        for lam in LAMBDAS:
            beta = ridge_fit(Xtr, ytr, lam)
            pv = Xva @ beta * std_safe[j] + mean[j]
            mae = np.abs(pv - data["val"]["y"][:, j])[Ev].mean()
            if best is None or mae < best[0]:
                best = (mae, lam, beta, pv)
        _, lam, beta, pv = best
        chosen[str(nodes[j])] = lam
        ridge_pred["val"][:, j] = pv
        Xte = node_features(zn, mask, data["test"]["t"], j)
        ridge_pred["test"][:, j] = Xte @ beta * std_safe[j] + mean[j]

    # 고수위 임계값: 학습 구간(인덱스 < train 끝) 원본 관측값만
    tr_vals = np.where(mask[:bounds["train"][1]], values[:bounds["train"][1]], np.nan)
    thr = {q: np.nanquantile(tr_vals, q, axis=0) for q in HIGH_Q}

    overall, by_node = [], []
    for split in ("val", "test"):
        d = data[split]
        preds = {"persistence": d["persist"], "node_ridge_ar": ridge_pred[split]}
        subsets = {"all": d["E"], **{f"high_q{int(q * 100)}": d["E"] & (d["y"] >= thr[q]) for q in HIGH_Q}}
        for model, p in preds.items():
            for sub, E in subsets.items():
                overall.append({"horizon_min": h, "split": split, "model": model, "subset": sub,
                                **metrics(np.nan_to_num(p), d["y"], E)})
                if split == "test":
                    for j, n in enumerate(nodes):
                        by_node.append({"horizon_min": h, "model": model, "subset": sub, "node_id": n,
                                        **metrics(np.nan_to_num(p[:, j]), d["y"][:, j], E[:, j])})
    te = data["test"]
    np.savez_compressed(OUT_PRED[h], t_idx=te["t"], y=te["y"].astype(np.float32), y_mask=te["ym"],
                        eval_mask=te["E"], persistence=te["persist"].astype(np.float32),
                        node_ridge_ar=ridge_pred["test"].astype(np.float32), node_ids=np.array(nodes, dtype=str))
    info = {"n_samples": {k: len(v["t"]) for k, v in data.items()},
            "pairs_target_observed": {k: int(v["ym"].sum()) for k, v in data.items()},
            "pairs_eval": {k: int(v["E"].sum()) for k, v in data.items()},
            "lambda": chosen, "thr": thr}
    return s, overall, by_node, info


def build_report(verif, infos, overall, by_node, nodes, elapsed) -> str:
    L = []
    w = L.append
    w("[강남] 데이터셋 검증 및 기준 모델 평가 리포트 (GNN 학습 없음)")
    w("=" * 66)
    w("1. 데이터셋 재검증 (샘플 CSV 플래그를 원본 마스크에서 독립 재계산)")
    for h, (v, tab) in verif.items():
        w(f"  [h={h}] " + ", ".join(f"{k}={x}" for k, x in v.items()))
        w("    " + tab.to_string().replace("\n", "\n    "))
    w("  보고서의 '199,783'은 리포트·샘플 CSV 어디에도 없음. h=30 test strict_raw는 19,696 (≤ 전체 19,783).")

    w("\n2. 입력·정답 규칙")
    w("  입력 x_value = 학습통계 정규화 값, 결측은 0 대체 / x_obs = 원본 관측 마스크 (대체값 식별). NaN 입력 없음.")
    w("  정답 = 원본 관측값만. 손실·지표는 정답 마스크가 1인 칸만 평균, 유효 정답 0개면 손실 0.")
    w("  공통 평가쌍 = 정답 원본 관측 AND 입력 창(60분)에 그 노드 관측 ≥1 (persistence 정의 가능 조건). 두 모델 동일.")
    for h, info in infos.items():
        w(f"  [h={h}] 샘플 {info['n_samples']}, 정답 관측쌍 {info['pairs_target_observed']}, "
          f"공통 평가쌍 {info['pairs_eval']}")
    w(f"  고수위 부분집합: 노드별 학습 구간 관측값 {', '.join(f'{q:.0%}' for q in HIGH_Q)} 분위수 이상인 정답 (사전 고정)")

    w("\n3. 결과 — 전체 (단위 m)")
    o = pd.DataFrame(overall)
    for h in HORIZONS:
        w(f"\n  [h={h}분]")
        t = o[o.horizon_min == h].pivot_table(index=["split", "subset"], columns="model",
                                              values=["MAE", "RMSE", "n_pairs"])
        w("    " + t.to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n    "))
    w("\n4. 노드별 test MAE (all) — 단위 m")
    b = pd.DataFrame(by_node)
    for h in HORIZONS:
        t = b[(b.horizon_min == h) & (b.subset == "all")].pivot(index="node_id", columns="model", values="MAE")
        t["ridge_better"] = t["node_ridge_ar"] < t["persistence"]
        w(f"  [h={h}분] ridge가 persistence보다 나은 노드 {int(t.ridge_better.sum())}/{len(t)}")
        w("    " + t.to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n    "))
    w("\n5. 선택된 λ (val MAE 기준)")
    for h, info in infos.items():
        w(f"  [h={h}] {info['lambda']}")
    w(f"\n처리 시간 {elapsed:,.0f}초")
    w("※ 기준 모델 결과일 뿐 GNN 성능 주장 아님. 그래프는 도로망 근접 근사이며 실제 하수관로 연결이 아니다.")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.perf_counter()
    values, mask, nodes, mean, std = load_base()
    bounds = split_bounds(len(values))
    verif, infos, overall, by_node = {}, {}, [], []
    for h in HORIZONS:
        s = load_samples(h)
        v = verify_samples(s, mask, h)
        g = s.groupby("split")
        tab = pd.DataFrame({"샘플": g.size(), "strict_raw": g.strict_raw.sum(), "strict_locf5": g.strict_locf5.sum(),
                            "masked": g.masked_any_target.sum(), "평가쌍": g.n_target_observed.sum()}
                           ).reindex(["train", "val", "test", "purged"])
        verif[h] = (v, tab)
        if not all(v.values()):
            raise SystemExit(f"데이터셋 검증 실패 h={h}: {v}")
        _, o, bn, info = run_horizon(h, values, mask, nodes, mean, std, bounds)
        overall += o
        by_node += bn
        infos[h] = info
    pd.DataFrame(overall).to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    pd.DataFrame(by_node).to_csv(OUT_NODE, index=False, encoding="utf-8-sig")
    report = build_report(verif, infos, overall, by_node, nodes, time.perf_counter() - t0)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
