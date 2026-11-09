"""T5 사건 초반 정보로 센서별 사건 최대 상승 예측. 규칙은 이 docstring 에 결과를 보기 전에 고정했다(2026-11-09). 이후 변경 금지. 바꾸려면 새 파일로 별도 실험을 만들고 "사후"로 표시한다.

==================================================================================================================================
[목적] 지금까지 센서 상태를 맞히는 문제는 단순 기준을 넘지 못했다(docs/19·22·23). 가장 강한 신호는 '사건 총강우 ↔ 센서 최대 상승'(docs/21)이다.
  이를 예측 문제로 바꾼다: 비 사건이 시작된 지 ORIGIN_H 시간이 지난 시점의 정보로, 그 사건에서 센서가 도달할 최대 상승(mm)을 예측한다. 두 시즌(2026 학습 → 2025 검증)으로 평가한다.
[단위] (사건, 센서). 최대 상승(목표) = 사건 시작 ~ 종료 + 3시간 창 안 센서의 최대 상승폭(mm, 지도와 같은 정의, docs/16). 창 안 유효 블록 6개 미만이면 제외.
[센서] 2025 센서 18개 ∩ 2026 센서 중 기록 안정(불안정 아님)인 것. 두 해에 같은 번호(23-xxxx)로 있는 센서만 쓴다(센서별 효과를 쓰기 위함).
[사건] 2026: 평가 이벤트 13개(prepare_rainfall_2026 의 is_eval). 2025: 이벤트 10개(gangnam_rain_events.csv 전부). 사건 시작 후 ORIGIN_H 시간이 사건 종료 + 3시간 안에 들어오지 않으면 제외.
[기점] τ = 사건 시작 + ORIGIN_H (주 2시간, 참고 1시간·3시간). 입력은 τ 까지의 정보만이다.
[입력] R = 사건 시작 ~ τ 의 강남 강우량계 평균 10분 강우의 합(mm). a = τ 블록의 상승폭(관측이 없으면 직전 3블록 안 마지막 관측, 그래도 없으면 제외). m = 사건 시작 ~ τ 의 최대 상승폭.
[모델 — 하이퍼파라미터 탐색 없음]
  B0 센서별 과거 중앙값: 학습 사건에서 그 센서의 최대 상승 중앙값. (참고)
  B1 지금까지의 최대 유지: 예측 = m.
  B2 max(B0, m): 센서 과거 중앙값과 이미 도달한 최대 중 큰 값. (주 비교 기준)
  M  릿지 회귀(sklearn Ridge, alpha = 1.0): 목표 = 최대 상승/100, 특징 = [센서 원핫, 센서 원핫 x log1p(R), log1p(R), a/100, m/100]. 예측 = max(모델 출력 x 100, m).
[평가] (1) 주: 2026 사건 전부로 학습 → 2025 사건으로 검증. (2) 참고: 2026 사건 하나 빼기(leave-one-event-out).
  지표 = 사건 안에서 센서 평균 절대오차(mm)를 구한 뒤 사건 평균(매크로). 부트스트랩: 평가 사건 복원추출 2000회, 시드 0, 차이 M − 기준의 95% 구간.
[판정 — 이 둘만, 주 평가(2025 검증)에서] M − B1, M − B2. 상한 < 0 이면 "M이 낫다", 하한 > 0 이면 "M이 나쁘다", 0 포함이면 "구분 불가".
  **둘 다 "M이 낫다"여야 '사건 초반 정보로 최대 상승이 예측된다'고 말한다.** 그렇지 않으면 단순 기준을 확실히 넘지 못했다고 보고한다.
[참고 보고(판정 아님)] B0 와의 차이, 단계(평상/상승/큰/매우 큰) 일치율과 한 단계 이내 비율, ORIGIN_H = 1·3 시간, leave-one-event-out.
[한계] 사건 13개(학습) + 10개(검증), 두 시즌. 위험이 아니라 평소 대비 상승이다. 강남 평균 강우라 비의 공간 차이는 반영하지 못한다.
==================================================================================================================================
실행: python -m src.analysis.t5_peak      출력: data/processed/gangnam_t5_peak_report.txt, gangnam_t5_peak_ci.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.analysis.rise_levels import STAGE_NAMES, rise_mm, stage
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import RAIN_NPZ as RAIN_2025_NPZ, build_inputs as inputs_2025
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import GANGNAM

OUT_REPORT = PROCESSED / "gangnam_t5_peak_report.txt"
OUT_CI = PROCESSED / "gangnam_t5_peak_ci.csv"
ORIGIN_MAIN, ORIGIN_REF = 2.0, (1.0, 3.0)
TAIL = pd.Timedelta(hours=3)
MIN_BLOCKS = 6
ALPHA = 1.0
MODELS = ("B0", "B1", "B2", "M")


def gn_rain_2025(bt) -> np.ndarray:
    z = np.load(RAIN_2025_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gn = [i for i, g in enumerate(z["gauge_gu"]) if g == GANGNAM]
    pos = rt.get_indexer(bt)
    with np.errstate(all="ignore"):
        return np.nanmean(z["values"][pos][:, gn], axis=1)


def pairs(bt, events, rise, rain, ids, origin_h: float) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        end = r.end + TAIL
        tau_t = r.start + pd.Timedelta(hours=origin_h)
        w = np.flatnonzero((bt >= r.start) & (bt <= end))
        pre = np.flatnonzero((bt >= r.start) & (bt <= tau_t))
        if not len(pre) or tau_t > end:
            continue
        tau = pre[-1]
        R = float(np.nansum(rain[pre]))
        for j, n in enumerate(ids):
            x = rise[w, j]
            if np.isfinite(x).sum() < MIN_BLOCKS:
                continue
            cur = rise[max(pre[0], tau - 3): tau + 1, j]
            cur = cur[np.isfinite(cur)]
            if not len(cur) or not np.isfinite(rise[pre, j]).any():
                continue
            rows.append(dict(event=r.event_id, node=n, R=R, a=float(cur[-1]), m=float(np.nanmax(rise[pre, j])), peak=float(np.nanmax(x))))
    return pd.DataFrame(rows)


def design(d: pd.DataFrame, ids: list[str]) -> np.ndarray:
    S = len(ids)
    oh = np.zeros((len(d), S))
    oh[np.arange(len(d)), d.node.map({n: i for i, n in enumerate(ids)}).to_numpy()] = 1.0
    lr = np.log1p(d.R.to_numpy())[:, None]
    return np.hstack([oh, oh * lr, lr, d.a.to_numpy()[:, None] / 100, d.m.to_numpy()[:, None] / 100])


def predict(train: pd.DataFrame, test: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    out = test.copy()
    med = train.groupby("node").peak.median()
    out["B0"] = out.node.map(med).fillna(train.peak.median())
    out["B1"] = out.m
    out["B2"] = np.maximum(out.B0, out.m)
    model = Ridge(alpha=ALPHA).fit(design(train, ids), train.peak.to_numpy() / 100)
    out["M"] = np.maximum(model.predict(design(test, ids)) * 100, out.m)
    return out


def event_mae(df: pd.DataFrame) -> pd.DataFrame:
    """사건 x 모델: 센서 평균 절대오차(mm)."""
    return pd.DataFrame({m: (df[m] - df.peak).abs().groupby(df.event).mean() for m in MODELS})


def boot_diff(e: pd.DataFrame, a: str, b: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    x = (e[a] - e[b]).to_numpy()
    d = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    return "M이 낫다" if hi < 0 else ("M이 나쁘다" if lo > 0 else "구분 불가")


def stage_rates(df: pd.DataFrame, m: str) -> tuple[float, float]:
    a, p = stage(df.peak.to_numpy()), stage(df[m].to_numpy())
    return float(np.mean(a == p)), float(np.mean(np.abs(a - p) <= 1))


def main() -> None:
    from src.dashboard.data import load
    D = load()
    i25 = inputs_2025()
    ids25, ids26 = i25["nodes"], D["nodes"]
    stable26 = set(D["info"].node_id[~D["info"].unstable])
    ids = [n for n in ids25 if n in set(ids26) and n in stable26]
    r25 = rise_mm({"blk": i25["blk"], "how": i25["how"]})[:, [ids25.index(n) for n in ids]]
    r26 = D["rise"][:, [ids26.index(n) for n in ids]]
    rain25, rain26 = gn_rain_2025(i25["bt"]), D["rain_gn"]
    ev26, ev25 = D["events"], i25["events"]
    L = ["[강남] T5 사건 초반 정보로 센서별 사건 최대 상승 예측 (규칙은 결과 보기 전 고정: src/analysis/t5_peak.py)", "=" * 80,
         f"센서 {len(ids)}개(2025·2026 공통, 2026 기록 안정). 2026 사건 {len(ev26)}개 학습, 2025 사건 {len(ev25)}개 검증. 부트스트랩 사건 {N_BOOT}회, 시드 {BOOT_SEED}.",
         "상승폭은 평소 대비(docs/16)이며 위험이 아니다. 모델 = 센서별 효과를 가진 릿지 회귀 하나(탐색 없음).", ""]
    csv = []
    verdicts = {}
    for h in (ORIGIN_MAIN, *ORIGIN_REF):
        p26 = pairs(D["bt"], ev26, r26, rain26, ids, h)
        p25 = pairs(i25["bt"], ev25, r25, rain25, ids, h)
        if p26.empty or p25.empty:
            L.append(f"[기점 사건 시작 {h:g}시간 후] 표본 없음")
            continue
        te = predict(p26, p25, ids)
        e = event_mae(te)
        tag = "주" if h == ORIGIN_MAIN else "참고"
        L.append(f"[기점 = 사건 시작 {h:g}시간 후 ({tag})] 학습 (사건,센서) {len(p26)}쌍({p26.event.nunique()}사건) → 검증 {len(p25)}쌍({p25.event.nunique()}사건)")
        L.append("  모델            사건평균 MAE(mm)   단계 일치 / 한 단계 이내")
        names = {"B0": "B0 센서 과거 중앙값", "B1": "B1 지금까지 최대", "B2": "B2 max(B0, B1)", "M": "M  릿지 회귀"}
        for m in MODELS:
            ex, one = stage_rates(te, m)
            L.append(f"  {names[m]:<18} {e[m].mean():8.1f}          {ex:.0%} / {one:.0%}")
            csv.append({"stat": f"H{h:g}_val2025_mae_{m}", "point": e[m].mean(), "lo": np.nan, "hi": np.nan})
        for base in ("B1", "B2", "B0"):
            d, lo, hi = boot_diff(e, "M", base)
            v = verdict(lo, hi)
            if h == ORIGIN_MAIN and base != "B0":
                verdicts[base] = v
            lab = "판정" if (h == ORIGIN_MAIN and base != "B0") else "참고"
            L.append(f"  {lab} MAE(M) − MAE({base}) = {d:+.1f} mm [{lo:+.1f}, {hi:+.1f}] → {v}")
            csv.append({"stat": f"H{h:g}_val2025_M_minus_{base}", "point": d, "lo": lo, "hi": hi})
        if h == ORIGIN_MAIN:                                                 # 참고: 2026 leave-one-event-out
            parts = []
            for ev in p26.event.unique():
                parts.append(predict(p26[p26.event != ev], p26[p26.event == ev], ids))
            loe = event_mae(pd.concat(parts, ignore_index=True))
            L.append("  [참고: 2026 사건 하나 빼기] 사건평균 MAE " + ", ".join(f"{m} {loe[m].mean():.1f}" for m in MODELS)
                     + " | " + ", ".join(f"M−{b} {boot_diff(loe, 'M', b)[0]:+.1f} [{boot_diff(loe, 'M', b)[1]:+.1f}, {boot_diff(loe, 'M', b)[2]:+.1f}]" for b in ("B1", "B2")))
        L.append("")
    ok = len(verdicts) == 2 and all(v == "M이 낫다" for v in verdicts.values())
    L += ["[결론 규칙] M − B1, M − B2 가 모두 'M이 낫다'여야 '사건 초반 정보로 센서별 최대 상승이 예측된다'고 말한다.",
          "  → " + ("두 시즌 검증에서 단순 기준보다 확실히 낫다(탐색 범위 안에서)." if ok else "단순 기준을 확실히 넘지 못했다. 예측이 된다고 보고하지 않는다."),
          "※ 사건 13개(학습)·10개(검증), 두 시즌. 강남 평균 강우. 위험이 아니라 평소 대비 상승."]
    pd.DataFrame(csv).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.4f")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
