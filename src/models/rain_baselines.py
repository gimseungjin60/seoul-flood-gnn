"""강우 입력이 수위 예측(T1)에 도움이 되는가: 노드별 Ridge AR 에 강우 특징만 더해 같은 조건에서 비교 (GNN 아님).

실행: python -m src.models.rain_baselines
출력: data/processed/gangnam_rain_baseline_metrics.csv, gangnam_rain_baseline_report.txt

사전 고정 규칙 (결과를 보기 전에 정함. 결과를 본 뒤 바꾸지 않는다)
  비교: 같은 샘플(gangnam_forecast_samples_h*.csv), 같은 평가쌍(정답 원본 관측 AND 입력 창에 그 노드 관측 >= 1),
        노드별 Ridge, λ는 val MAE로 노드마다 선택(기존 baselines.py와 동일 격자). 기존 node_ridge_ar(강우 없음)와
        '강우 특징만 다른' 모델을 비교한다. persistence는 참고.
  강우 특징(예측 시점 t에서 구간이 끝난 슬롯만: 슬롯 끝 분 <= t-1): 최근 6개 슬롯(60분)의 log1p(mm), 직전 6개 슬롯 중 결측 수,
        선행 누적 3시간·6시간·24시간 log1p(mm). 결측 슬롯은 0mm로 두고 결측 수 특징으로 표시한다. train 통계로 표준화.
  강우 원천 두 가지: nearest(센서별 최근접 강우량계) / gangnam_mean(강남 3개소 평균).
  평가 부분집합: all / high_q95 / high_q99(노드별 train 분위수 이상 정답) / event(정답 시각이 유의 강우 이벤트 시작~끝+3시간).
  판정: '강우가 도움이 된다' = test all MAE가 5% 이상 개선 AND test event MAE가 10% 이상 개선 AND val에서도 같은 방향(all, event 모두 개선).
        한 쪽이라도 못 미치면 '도움이 일관되지 않음'. 사후에 기준을 바꾸지 않는다.
한계: 선형 모델이라 강우-수위의 비선형 반응은 담지 못한다(비선형 모델은 별도). 예보 강우는 쓰지 않는다(실측만).
"""
import time

import numpy as np
import pandas as pd

from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV
from src.data.build_forecast_dataset import split_bounds
from src.data.clean_nodes import PROCESSED
from src.data.forecast_inputs import HORIZONS, load_base, load_samples, make_targets, metrics, normalize
from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ
from src.models.baselines import HIGH_Q, LAMBDAS, eval_mask, last_observed, node_features, ridge_fit

OUT_METRICS = PROCESSED / "gangnam_rain_baseline_metrics.csv"
REPORT = PROCESSED / "gangnam_rain_baseline_report.txt"
N_SLOTS = 6
CUM_SLOTS = {"cum3h": 18, "cum6h": 36, "cum24h": 144}
EVENT_AFTER_H = 3
THR_ALL, THR_EVENT = 0.05, 0.10


def rain_features(rain: np.ndarray, label_min: np.ndarray, t: np.ndarray) -> np.ndarray:
    """rain (S,) mm/10분(NaN=결측), label_min (S,) 슬롯 끝 분(수위 분 인덱스 기준), t (n,) anchor 분 인덱스.
    t 시점에 이미 끝난 슬롯(label <= t-1)만 사용. 반환 (n, N_SLOTS + 1 + 3)."""
    k = np.searchsorted(label_min, t - 1, side="right") - 1            # 마지막으로 끝난 슬롯
    if (k < max(CUM_SLOTS.values()) - 1).any():
        raise ValueError("선행 강우(24시간) 계산에 필요한 이전 슬롯이 부족하다 (5월 자료 확인)")
    r0 = np.nan_to_num(rain, nan=0.0)
    miss = np.isnan(rain).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(r0)])
    recent = np.stack([r0[k - j] for j in range(N_SLOTS)], axis=1)
    n_miss = sum(miss[k - j] for j in range(N_SLOTS))[:, None]
    cum = np.stack([cs[k + 1] - cs[k + 1 - n] for n in CUM_SLOTS.values()], axis=1)
    return np.hstack([np.log1p(recent), n_miss, np.log1p(cum)])


def standardize(train_x: np.ndarray):
    mu, sd = train_x.mean(0), train_x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return lambda x: (x - mu) / sd


def event_mask_for_targets(t_target: np.ndarray, s0: pd.Timestamp, events: pd.DataFrame) -> np.ndarray:
    ts = s0 + pd.to_timedelta(t_target, unit="min")
    m = np.zeros(len(t_target), dtype=bool)
    for e in events.itertuples():
        m |= (ts >= e.start) & (ts <= e.end + pd.Timedelta(hours=EVENT_AFTER_H))
    return m


def run_horizon(h, values, mask, nodes, mean, std, bounds, rain_sources, label_min, s0, events):
    s = load_samples(h)
    std_safe = np.where(std > 0, std, 1.0)
    zn = np.where(mask, normalize(values, mean, std), 0.0)
    parts = {n: s.loc[s["split"] == n, "t_idx"].to_numpy() for n in ("train", "val", "test")}
    data = {}
    for n, t in parts.items():
        y, ym = make_targets(values, mask, t, h)
        p = last_observed(values, mask, t)
        data[n] = dict(t=t, y=y, ym=ym, persist=p, E=eval_mask(ym, p),
                       ev=event_mask_for_targets(t + h, s0, events))
    tr_vals = np.where(mask[:bounds["train"][1]], values[:bounds["train"][1]], np.nan)
    thr = {q: np.nanquantile(tr_vals, q, axis=0) for q in HIGH_Q}

    preds = {sp: {"persistence": data[sp]["persist"]} for sp in ("val", "test")}
    chosen = {}
    for variant in ("none", "nearest", "gangnam_mean"):
        pv_all = {sp: np.full(data[sp]["y"].shape, np.nan) for sp in ("val", "test")}
        for j in range(len(nodes)):
            feats = {sp: node_features(zn, mask, data[sp]["t"], j) for sp in ("train", "val", "test")}
            if variant != "none":
                r = rain_sources[variant][j]
                rf = {sp: rain_features(r, label_min, data[sp]["t"]) for sp in feats}
                rows_tr = data["train"]["ym"][:, j]
                f = standardize(rf["train"][rows_tr])
                feats = {sp: np.hstack([feats[sp][:, :-1], f(rf[sp]), feats[sp][:, -1:]]) for sp in feats}
            tr = data["train"]
            rows = tr["ym"][:, j]
            Xtr, ytr = feats["train"][rows], normalize(tr["y"][rows, j], mean[j], std[j])
            Ev = data["val"]["E"][:, j]
            best = None
            for lam in LAMBDAS:
                beta = ridge_fit(Xtr, ytr, lam)
                pv = feats["val"] @ beta * std_safe[j] + mean[j]
                mae = np.abs(pv - data["val"]["y"][:, j])[Ev].mean()
                if best is None or mae < best[0]:
                    best = (mae, lam, beta, pv)
            _, lam, beta, pv = best
            chosen[(variant, str(nodes[j]))] = lam
            pv_all["val"][:, j] = pv
            pv_all["test"][:, j] = feats["test"] @ beta * std_safe[j] + mean[j]
        for sp in ("val", "test"):
            preds[sp][f"ridge_{variant}"] = pv_all[sp]

    rows, by_node = [], []
    for sp in ("val", "test"):
        d = data[sp]
        subsets = {"all": d["E"], **{f"high_q{int(q * 100)}": d["E"] & (d["y"] >= thr[q]) for q in HIGH_Q},
                   "event": d["E"] & d["ev"][:, None]}
        for model, p in preds[sp].items():
            for sub, E in subsets.items():
                rows.append({"horizon_min": h, "split": sp, "model": model, "subset": sub, **metrics(np.nan_to_num(p), d["y"], E)})
                if sp == "test" and sub in ("all", "event"):
                    for j, nid in enumerate(nodes):
                        by_node.append({"horizon_min": h, "model": model, "subset": sub, "node_id": nid,
                                        **metrics(np.nan_to_num(p[:, j]), d["y"][:, j], E[:, j])})
    return rows, by_node, chosen


def verdict(m: pd.DataFrame, h: int, variant: str) -> dict:
    def mae(sp, sub, model):
        r = m[(m.horizon_min == h) & (m.split == sp) & (m.subset == sub) & (m.model == model)]
        return float(r.MAE.iloc[0])
    base = "ridge_none"
    new = f"ridge_{variant}"
    gain = {(sp, sub): 1 - mae(sp, sub, new) / mae(sp, sub, base) for sp in ("val", "test") for sub in ("all", "event")}
    ok = (gain[("test", "all")] >= THR_ALL and gain[("test", "event")] >= THR_EVENT
          and gain[("val", "all")] > 0 and gain[("val", "event")] > 0)
    return {"h": h, "variant": variant, **{f"{sp}_{sub}": g for (sp, sub), g in gain.items()}, "helps": bool(ok)}


def main() -> None:
    t0 = time.perf_counter()
    values, mask, nodes, mean, std = load_base()
    bounds = split_bounds(len(values))
    s = np.load(SEWER_NPZ, allow_pickle=True)
    s0 = pd.Timestamp(s["times"][0])
    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    label_min = ((rt - s0) / pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    gids = [str(g) for g in z["gauge_ids"]]
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    rv = z["values"].astype(np.float64)
    gn = [i for i, u in enumerate(z["gauge_gu"]) if u == "강남구"]
    with np.errstate(all="ignore"):
        mean_gn = np.nanmean(rv[:, gn], axis=1)
    sources = {"nearest": [rv[:, gids.index(asg.loc[n, "gauge_id"])] for n in nodes],
               "gangnam_mean": [mean_gn for _ in nodes]}
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])

    rows, by_node, chosen = [], [], {}
    for h in HORIZONS:
        r, b, c = run_horizon(h, values, mask, nodes, mean, std, bounds, sources, label_min, s0, events)
        rows += r
        by_node += b
        chosen[h] = c
    m = pd.DataFrame(rows)
    m.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    bn = pd.DataFrame(by_node)

    L = []
    w = L.append
    w("[강남] 강우 입력 효과 기준 모델 리포트 (노드별 Ridge AR + 강우 특징, GNN 아님)")
    w("=" * 70)
    w("※ 사전 고정 규칙은 src/models/rain_baselines.py 상단 docstring 참조(결과 확인 전 작성). 같은 샘플·같은 평가쌍·같은 λ 선택 규칙.")
    w(f"  유의 강우 이벤트 {len(events)}개, event 부분집합 = 정답 시각이 이벤트 시작~끝+{EVENT_AFTER_H}시간")
    for h in HORIZONS:
        w(f"\n[h={h}분] 전체/부분집합별 MAE, RMSE (m)")
        t = m[m.horizon_min == h].pivot_table(index=["split", "subset"], columns="model", values=["MAE", "RMSE", "n_pairs"], sort=False)
        t = t.reindex(columns=["persistence", "ridge_none", "ridge_nearest", "ridge_gangnam_mean"], level=1)
        w("  " + t["MAE"].to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n  "))
        w("  RMSE:\n  " + t["RMSE"].to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n  "))
        w("  평가쌍 수(persistence 열 기준): " + ", ".join(f"{a}/{b}={int(v)}" for (a, b), v in t["n_pairs"]["persistence"].items()))
    w("\n판정 (ridge_none 대비 MAE 개선율, 사전 규칙)")
    vs = [verdict(m, h, v) for h in HORIZONS for v in ("nearest", "gangnam_mean")]
    w(pd.DataFrame(vs).to_string(index=False, float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)))
    w(f"  기준: test all >= {THR_ALL:.0%} AND test event >= {THR_EVENT:.0%} AND val all, event 모두 > 0")
    w("\n노드별 test event MAE (m), ridge_none vs ridge_nearest, h=10분")
    for h in HORIZONS:
        t = bn[(bn.horizon_min == h) & (bn.subset == "event")].pivot(index="node_id", columns="model", values="MAE")
        t["n"] = bn[(bn.horizon_min == h) & (bn.subset == "event") & (bn.model == "ridge_none")].set_index("node_id")["n_pairs"]
        t["nearest_better"] = t["ridge_nearest"] < t["ridge_none"]
        w(f"  [h={h}분] 강우(최근접)가 나은 노드 {int(t.nearest_better.sum())}/{len(t)}")
        w("    " + t[["persistence", "ridge_none", "ridge_nearest", "ridge_gangnam_mean", "n", "nearest_better"]].to_string(
            float_format=lambda x: f"{x:.4f}").replace("\n", "\n    "))
    w("\n주의")
    w("  - 선형 모델이다. 강우-수위의 비선형 반응(임계 이후 급상승 등)은 담지 못해 강우의 효과가 과소평가될 수 있다.")
    w("  - 강우 입력은 실측만이며 예보가 아니다. 수위가 강우에 10분 안팎으로 반응하므로 h=30분의 강우 이득은 작을 수 있다.")
    w("  - test의 이벤트는 2개(약한 1개 + 45mm 1개)다. event 지표는 소수 사건에 의존하고 평가쌍은 서로 독립이 아니다.")
    w(f"처리 시간 {time.perf_counter() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
