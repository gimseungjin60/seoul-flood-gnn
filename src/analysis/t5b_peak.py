"""T5b: T5(src/analysis/t5_peak.py, docs/24)의 추가 평가. 결과를 보기 전에 규칙을 고정했다(2026-11-09). 이후 변경 금지.
**주의: 이 평가는 T5 의 결과(점추정은 모델이 낫지만 구간이 0 을 포함)를 본 뒤에 추가한 것이다.** 그래서 아래 네 평가를 결과와 관계없이 모두 보고하고,
T5 의 주 판정(2026 학습 → 2025 검증)은 이미 '구분 불가'였음을 함께 적는다. 모델·특징·기준·지표·부트스트랩은 T5 와 같다(변경 없음).

[평가 — 이 네 가지를 모두 보고한다]
  A  방향 바꾸기: 2025 사건으로 학습 → 2026 사건으로 검증. 강우 = 강남 평균.
  B  (이번 묶음의 주 판정) 두 해 사건을 합쳐 leave-one-event-out(사건 하나를 빼고 나머지로 학습, 뺀 사건을 평가). 강우 = 강남 평균.
  C  B 와 같지만 강우 = 그 센서의 최근접 강우량계 10분 강우(센서별, T1·T2 와 같은 배정 파일).
  D  T5 의 방향(2026 학습 → 2025 검증)에서 강우 = 센서별 최근접 강우량계.
[판정 — B 하나] B 에서 MAE(M) − MAE(B1), MAE(M) − MAE(B2) 의 사건 단위 부트스트랩(2000회, 시드 0) 95% 구간이 둘 다 상한 < 0 이면 "M이 낫다".
  둘 다 "M이 낫다"일 때만 '두 시즌을 합친 평가에서 단순 기준보다 확실히 낫다'고 말한다. 그래도 T5 의 사전 주 판정이 통과하지 못했다는 사실과
  평가를 추가했다는 사실을 함께 적는다. A, C, D 는 참고(판정 아님)이며 방향의 일관성만 본다.
[기점] 사건 시작 2시간 후(T5 와 같음). 사건 키 = (연도, event_id).
[한계] 사건 수(2025 10개 중 2시간 조건을 만족하는 8개 + 2026 13개), 두 시즌. 사후 추가 평가.
실행: python -m src.analysis.t5b_peak      출력: data/processed/gangnam_t5b_peak_report.txt, gangnam_t5b_peak_ci.csv
"""
import numpy as np
import pandas as pd

from src.analysis.rise_levels import rise_mm
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_loso import ASSIGN_CSV, RAIN_NPZ as RAIN_2025_NPZ, build_inputs as inputs_2025
from src.analysis.t5_peak import MIN_BLOCKS, MODELS, ORIGIN_MAIN, TAIL, boot_diff, event_mae, gn_rain_2025, predict, stage_rates, verdict
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_GAUGE, OUT_NPZ as RAIN_2026_NPZ

OUT_REPORT = PROCESSED / "gangnam_t5b_peak_report.txt"
OUT_CI = PROCESSED / "gangnam_t5b_peak_ci.csv"


def sensor_rain(npz, assign_csv, bt, ids) -> np.ndarray:
    """(B, J) 센서별 최근접 강우량계 10분 강우(mm). 수위 블록 끝 라벨과 같은 슬롯."""
    z = np.load(npz, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    pos = rt.get_indexer(bt)
    asg = pd.read_csv(assign_csv, dtype={"gauge_id": str}).set_index("node_id")
    return np.stack([z["values"][pos, gids.index(asg.loc[n, "gauge_id"])].astype(float) for n in ids], axis=1)


def pairs(bt, events, rise, rain, ids, year: int, origin_h: float = ORIGIN_MAIN) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        end = r.end + TAIL
        tau_t = r.start + pd.Timedelta(hours=origin_h)
        w = np.flatnonzero((bt >= r.start) & (bt <= end))
        pre = np.flatnonzero((bt >= r.start) & (bt <= tau_t))
        if not len(pre) or tau_t > end:
            continue
        tau = pre[-1]
        for j, n in enumerate(ids):
            x = rise[w, j]
            if np.isfinite(x).sum() < MIN_BLOCKS:
                continue
            cur = rise[max(pre[0], tau - 3): tau + 1, j]
            cur = cur[np.isfinite(cur)]
            if not len(cur) or not np.isfinite(rise[pre, j]).any():
                continue
            R = float(np.nansum(rain[pre, j] if rain.ndim == 2 else rain[pre]))
            rows.append(dict(event=f"{year}-{r.event_id}", year=year, node=n, R=R, a=float(cur[-1]), m=float(np.nanmax(rise[pre, j])), peak=float(np.nanmax(x))))
    return pd.DataFrame(rows)


def loeo(df: pd.DataFrame, ids) -> pd.DataFrame:
    return pd.concat([predict(df[df.event != e], df[df.event == e], ids) for e in df.event.unique()], ignore_index=True)


def block(L, csv, title, te, tag) -> None:
    e = event_mae(te)
    L.append(f"[{title}] 평가 (사건,센서) {len(te)}쌍, 사건 {te.event.nunique()}개")
    L.append("  사건평균 MAE(mm): " + ", ".join(f"{m} {e[m].mean():.1f}" for m in MODELS) + " | M 단계 일치 {:.0%} / 한 단계 이내 {:.0%}".format(*stage_rates(te, "M")))
    out = {}
    for base in ("B1", "B2"):
        d, lo, hi = boot_diff(e, "M", base)
        out[base] = verdict(lo, hi)
        L.append(f"  MAE(M) − MAE({base}) = {d:+.1f} mm [{lo:+.1f}, {hi:+.1f}] → {out[base]}")
        csv.append({"stat": f"{tag}_M_minus_{base}", "point": d, "lo": lo, "hi": hi})
    L.append("")
    return out


def main() -> None:
    from src.dashboard.data import load
    D = load()
    i25 = inputs_2025()
    ids25, ids26 = i25["nodes"], D["nodes"]
    stable26 = set(D["info"].node_id[~D["info"].unstable])
    ids = [n for n in ids25 if n in set(ids26) and n in stable26]
    r25 = rise_mm({"blk": i25["blk"], "how": i25["how"]})[:, [ids25.index(n) for n in ids]]
    r26 = D["rise"][:, [ids26.index(n) for n in ids]]
    gn25, gn26 = gn_rain_2025(i25["bt"]), D["rain_gn"]
    sg25 = sensor_rain(RAIN_2025_NPZ, ASSIGN_CSV, i25["bt"], ids)
    sg26 = sensor_rain(RAIN_2026_NPZ, OUT_GAUGE, D["bt"], ids)
    ev25, ev26 = i25["events"], D["events"]
    p25g, p26g = pairs(i25["bt"], ev25, r25, gn25, ids, 2025), pairs(D["bt"], ev26, r26, gn26, ids, 2026)
    p25s, p26s = pairs(i25["bt"], ev25, r25, sg25, ids, 2025), pairs(D["bt"], ev26, r26, sg26, ids, 2026)
    L = ["[강남] T5b 사건 초반 정보로 센서별 최대 상승 예측: 추가 평가 (규칙은 결과 보기 전 고정: src/analysis/t5b_peak.py)", "=" * 80,
         f"센서 {len(ids)}개. 사건 수(2시간 조건 충족): 2025 {p25g.event.nunique()}개, 2026 {p26g.event.nunique()}개. 모델·기준·지표는 T5와 같다. 부트스트랩 사건 {N_BOOT}회, 시드 {BOOT_SEED}.",
         "※ 이 평가는 T5 결과(2026 학습 → 2025 검증에서 구분 불가)를 본 뒤 추가한 것이다. 네 평가를 모두 보고한다.", ""]
    csv = []
    a = block(L, csv, "A 2025 학습 → 2026 검증 (강남 평균 강우, 참고)", predict(p25g, p26g, ids), "A_2025to2026")
    b = block(L, csv, "B 두 해 합쳐 사건 하나 빼기 (강남 평균 강우, 이번 묶음의 주 판정)", loeo(pd.concat([p25g, p26g], ignore_index=True), ids), "B_pooled_loeo")
    c = block(L, csv, "C 두 해 합쳐 사건 하나 빼기 (센서별 최근접 강우량계, 참고)", loeo(pd.concat([p25s, p26s], ignore_index=True), ids), "C_pooled_loeo_sensor_gauge")
    d = block(L, csv, "D 2026 학습 → 2025 검증 (센서별 최근접 강우량계, 참고)", predict(p26s, p25s, ids), "D_2026to2025_sensor_gauge")
    ok = all(v == "M이 낫다" for v in b.values())
    L += ["[판정] B 에서 M − B1, M − B2 가 모두 'M이 낫다'여야 한다.",
          "  → " + ("두 시즌을 합친 평가에서 단순 기준보다 확실히 낫다. 단 T5 의 사전 주 판정(2026→2025)은 통과하지 못했고, 이 평가는 사후에 추가했다." if ok
                    else "두 시즌을 합쳐도 단순 기준을 확실히 넘지 못했다."),
          "  참고(A, C, D): M − B1 판정 " + ", ".join(f"{k} {v['B1']}" for k, v in (("A", a), ("C", c), ("D", d))) + " | M − B2 판정 "
          + ", ".join(f"{k} {v['B2']}" for k, v in (("A", a), ("C", c), ("D", d))),
          "※ 사건 수가 적고 두 시즌. 위험이 아니라 평소 대비 상승이다."]
    pd.DataFrame(csv).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.4f")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
