"""T3 경보형 평가 실행·판정: 사전 고정 규칙(src/analysis/t3_alert_rules.py)대로 S0~S2, L 의 '큰 상승 도달' 점수를 만들고 AP 를 비교한다.

실행: python -m src.analysis.t3_alert
출력: data/processed/gangnam2026_t3_alert_report.txt, gangnam2026_t3_alert_ci.csv
기점 t 의 특징은 t 까지의 정보만 쓴다. 학습은 purge 밖 구간만. 규칙은 결과를 보기 전에 고정했다.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.analysis.rise_levels import rise_mm
from src.analysis.t2_2026 import build_inputs, load_roles
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import event_blocks, fold_slices
from src.analysis.t3_alert_rules import (BIG_MM, C_LOGIT, FPR_Q, HORIZON_MAIN, HORIZON_REF, LABEL, MIN_FUTURE_FRAC, MODELS, predictable, verdict)
from src.data.clean_nodes import PROCESSED

OUT_REPORT = PROCESSED / "gangnam2026_t3_alert_report.txt"
OUT_CI = PROCESSED / "gangnam2026_t3_alert_ci.csv"
LAGS = (1, 3, 6)


def labels(a: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, J) 라벨 y 와 정의 여부. 기점 a[t] < BIG 이고 관측, 이후 H 블록 중 관측이 H*0.5 이상이어야 정의."""
    B, J = a.shape
    y, ok = np.zeros((B, J)), np.zeros((B, J), bool)
    for t in range(B - H):
        fut = a[t + 1: t + 1 + H]
        n_obs = np.isfinite(fut).sum(0)
        cur = a[t]
        valid = np.isfinite(cur) & (cur < BIG_MM) & (n_obs >= np.ceil(H * MIN_FUTURE_FRAC))
        ok[t] = valid
        y[t] = np.where(valid, (np.where(np.isfinite(fut), fut, -np.inf) >= BIG_MM).any(0), 0)
    return y, ok


def features(a: np.ndarray, rain_feat: list, time_feat: np.ndarray, tg: list[int]):
    """(B, J, F) 특징과 규칙 점수 S0, S1, S2 (B, J). t 까지의 정보만."""
    B, J = a.shape[0], len(tg)
    cur = a[:, :]
    diffs, miss = [], np.zeros((B, J))
    for k in LAGS:
        lag = np.full((B, J), np.nan)
        lag[k:] = cur[:-k]
        d = cur - lag
        miss += ~np.isfinite(d)
        diffs.append(np.nan_to_num(d) / 100)
    X = np.stack([np.nan_to_num(cur) / 100, *diffs, miss], axis=-1)
    rf = np.stack([rain_feat[j] for j in range(J)], axis=1)
    tf = np.repeat(time_feat[:, None, :], J, axis=1)
    F = np.concatenate([X, rf, tf], axis=-1)
    lag3 = np.full((B, J), np.nan)
    lag3[3:] = cur[:-3]
    s0 = cur
    s1 = cur + np.where(np.isfinite(cur - lag3), cur - lag3, 0.0)
    s2 = np.expm1(rf[:, :, :6]).sum(-1)
    return F, {"S0": s0, "S1": s1, "S2": s2}


def run(H: int, inp: dict, tg: list[int], rise: np.ndarray, log=print):
    a = rise[:, tg]
    B, J = a.shape
    y, ok = labels(a, H)
    F, S = features(a, [inp["rain_feat"][j] for j in tg], inp["time_feat"], list(range(J)))
    ev = event_blocks(inp["bt"], inp["events"])
    ev_of = np.full(B, -1)
    for eid, idx in ev[::-1]:                                              # 겹치면 먼저 나오는 이벤트에 귀속
        ev_of[idx] = eid
    rows = []
    for f, (eval_idx, train_idx, train_mask) in enumerate(fold_slices(B)):
        tr_t = np.array([t for t in train_idx if t + H < B and train_mask[t: t + H + 1].all()])
        tr = ok[tr_t]                                                      # (T, J)
        Xtr, ytr = F[tr_t][tr], y[tr_t][tr]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C_LOGIT, class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
        ev_t = eval_idx[(ev_of[eval_idx] >= 0)]
        te = ok[ev_t]
        if not te.any():
            continue
        Xte = F[ev_t][te]
        score = {"L": clf.predict_proba(sc.transform(Xte))[:, 1], **{m: S[m][ev_t][te] for m in ("S0", "S1", "S2")}}
        thr = {"L": np.quantile(clf.predict_proba(sc.transform(Xtr))[ytr == 0, 1], FPR_Q),
               **{m: np.quantile(S[m][tr_t][tr][ytr == 0], FPR_Q) for m in ("S0", "S1", "S2")}}
        tt, jj = np.nonzero(te)
        df = pd.DataFrame({"fold": f, "t": ev_t[tt], "j": jj, "event": ev_of[ev_t[tt]], "y": y[ev_t][te]})
        for m in MODELS:
            df[m] = score[m]
            df[m + "_alarm"] = score[m] > thr[m]
        rows.append(df)
        log(f"  H={H * 10}분 fold {f + 1}: 학습 {len(ytr):,} (양성 {int(ytr.sum()):,}), 평가 {len(df):,} (양성 {int(df.y.sum()):,})")
    return pd.concat(rows, ignore_index=True)


def ap(df: pd.DataFrame, m: str) -> float:
    return float(average_precision_score(df.y, df[m])) if df.y.sum() > 0 else float("nan")


def boot_diff(df: pd.DataFrame, base: str, rng: np.random.Generator) -> tuple[float, float, float]:
    point = ap(df, "L") - ap(df, base)
    groups = {e: g for e, g in df.groupby("event")}
    ids = np.array(list(groups))
    d = []
    for _ in range(N_BOOT):
        s = pd.concat([groups[e] for e in rng.choice(ids, len(ids))], ignore_index=True)
        if s.y.sum() == 0:
            continue
        d.append(average_precision_score(s.y, s["L"]) - average_precision_score(s.y, s[base]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    rise = rise_mm({"blk": inp["blk"], "how": inp["how"]})
    print(f"노드 {len(nodes)}개, 예측 대상 {len(tg)}개, 블록 {len(inp['bt']):,}개, 평가 이벤트 {len(inp['events'])}개", flush=True)
    res = {H: run(H, inp, tg, rise) for H in (HORIZON_MAIN, HORIZON_REF)}
    csv, L = [], ["[강남 2026] T3 경보형 평가: 이후 60분(참고 30분) 안에 '큰 상승'(200 mm 이상)에 도달하는가 (사전 고정 규칙: t3_alert_rules.py)",
                  "=" * 80, "상승폭은 평소 대비(docs/16)이며 위험이 아니다. 기점 상승폭이 200 mm 미만인 (시각, 센서)만 대상. 예측 대상 17개, 5폴드, 이벤트 창 안 표본만 평가.",
                  "학습 모델은 로지스틱 회귀 하나뿐(그래프·GNN 없음). 부트스트랩 이벤트 13개 2000회, 시드 0.", ""]
    verd = {}
    for H in (HORIZON_MAIN, HORIZON_REF):
        df = res[H]
        rng = np.random.default_rng(BOOT_SEED)
        tag = "주" if H == HORIZON_MAIN else "참고"
        L += [f"[H = {H * 10}분 ({tag})] 평가 표본 {len(df):,}개, 양성 {int(df.y.sum()):,}개 (기준 양성 비율 {df.y.mean():.1%}), "
              f"양성이 있는 (이벤트, 센서) {df[df.y == 1].groupby(['event', 'j']).ngroups}개"]
        L.append("  모델            AP     재현율@헛경보  (임계 = 학습 음성 점수 90% 분위, 실제 헛경보율)")
        for m in MODELS:
            pos, neg = df[df.y == 1], df[df.y == 0]
            rec, fa = pos[m + "_alarm"].mean(), neg[m + "_alarm"].mean()
            L.append(f"  {LABEL[m]:<12} {ap(df, m):6.3f}   {rec:6.1%} @ {fa:5.1%}")
            csv.append({"stat": f"H{H * 10}_ap_{m}", "point": ap(df, m), "lo": np.nan, "hi": np.nan})
        for tagp, base in (("P1", "S0"), ("P2", "S1"), ("P3", "S2")):
            p, lo, hi = boot_diff(df, base, rng)
            v = verdict(lo, hi)
            verd[(H, tagp)] = v
            L.append(f"  {tagp}{'(참고)' if H == HORIZON_REF else ''} AP(L) − AP({base}) = {p:+.3f} [{lo:+.3f}, {hi:+.3f}] → {v}")
            csv.append({"stat": f"H{H * 10}_{tagp}_L_minus_{base}", "point": p, "lo": lo, "hi": hi})
        L.append("")
    ok = predictable(verd[(HORIZON_MAIN, "P1")], verd[(HORIZON_MAIN, "P2")])
    L += ["[결론 규칙] P1과 P2가 모두 'L이 낫다'여야 '경보형으로 예측이 된다'고 말한다.",
          "  → " + ("경보형으로 단순 기준(현재 상승폭, 추세 외삽)보다 먼저 알아채는 것이 확인됨." if ok
                    else "단순 기준을 확실히 넘지 못했다. '경보형으로 예측이 된다'고 보고하지 않는다."),
          "※ 이벤트 13개, 한 시즌. 평소 수위 기준에 평가 구간이 들어 있다. 위험 판정이 아니다."]
    pd.DataFrame(csv).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.5f")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
