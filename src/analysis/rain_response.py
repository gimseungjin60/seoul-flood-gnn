"""센서별 강우-상승 반응(시나리오형 요약). 규칙은 결과를 보기 전에 이 docstring 과 상수로 고정했다(2026-11-09). 모델 학습 없음.

문제: "비가 이 정도 오면 이 센서는 과거에 얼마나 올랐나"를 센서마다 요약한다. 예측 성능을 주장하지 않는다.

고정 규칙
- 단위 (사건, 센서): 사건 시작 ~ 종료+3시간 창 안 센서 첨두 상승(mm, 지도와 같은 정의, docs/16). 2026 평가 이벤트 13개.
- 센서: 기록 안정(불안정 아님) 센서. 사건 창 안 유효 블록이 MIN_BLOCKS(6) 미만인 (사건, 센서)는 제외.
- 강우 = 사건 총강우량(강남 강우량계 평균, mm). 구간: 30 mm 미만 / 50 mm 이상 (30~50 mm 사건은 없었다).
- 센서별 보고: 사건 수 n, 스피어먼 순위상관 rho(총강우, 첨두), 사건 단위 부트스트랩 rho 95% 구간(N_BOOT=2000, 시드 0, 백분위),
  구간별 첨두 중앙값과 최소~최대, 사건 중 도달한 최고 단계 분포.
- 분류: n < MIN_EVENTS(8) → "자료 부족". 아니면 rho 구간 하한 > 0 → "강우와 함께 움직임", 아니면 "확인 안 됨".
- 보조(판정 아님): 사건 시작 전 24시간 강남 평균 강우(선행강우)가 총강우로 설명되지 않는 부분(순위 잔차)과 관련 있는지, 센서 중앙 첨두 기준으로 한 번만 본다.
- 결과를 보고 규칙·구간·임계를 바꾸지 않는다. 이벤트가 13개뿐이라 구간이 넓다.

실행: python -m src.analysis.rain_response
출력: data/processed/rain_response_by_sensor.csv, rain_response_by_event_sensor.csv, rain_response_report.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.analysis.rise_levels import STAGE_NAMES, stage
from src.data.clean_nodes import PROCESSED

OUT_SENSOR = PROCESSED / "rain_response_by_sensor.csv"
OUT_PAIRS = PROCESSED / "rain_response_by_event_sensor.csv"
OUT_REPORT = PROCESSED / "rain_response_report.txt"
TAIL = pd.Timedelta(hours=3)
MIN_BLOCKS, MIN_EVENTS, N_BOOT, BOOT_SEED = 6, 8, 2000, 0
HEAVY_MM, LIGHT_MM = 50, 30
ANTECEDENT = pd.Timedelta(hours=24)


def pairs(bt, events, rise, rain_gn, node_ids, unstable) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        w = np.flatnonzero((bt >= r.start) & (bt <= r.end + TAIL))
        ant = float(np.nansum(rain_gn[(bt >= r.start - ANTECEDENT) & (bt < r.start)]))
        for j, n in enumerate(node_ids):
            if unstable[j] or np.isfinite(rise[w, j]).sum() < MIN_BLOCKS:
                continue
            p = float(np.nanmax(rise[w, j]))
            rows.append(dict(event_id=r.event_id, node_id=n, total_mm=r.total_mm, antecedent_24h_mm=ant, peak_mm=p, stage=int(stage(p))))
    return pd.DataFrame(rows)


def rho_boot(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    rho = spearmanr(x, y)[0]
    n, b = len(x), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        b.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)
    return float(rho), float(lo), float(hi)


def by_sensor(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for n, g in d.groupby("node_id"):
        row = dict(node_id=n, n_events=len(g))
        light, heavy = g[g.total_mm < LIGHT_MM], g[g.total_mm >= HEAVY_MM]
        for tag, s in (("light", light), ("heavy", heavy)):
            row.update({f"{tag}_n": len(s), f"{tag}_median_mm": s.peak_mm.median() if len(s) else np.nan,
                        f"{tag}_min_mm": s.peak_mm.min() if len(s) else np.nan, f"{tag}_max_mm": s.peak_mm.max() if len(s) else np.nan})
        row["max_stage_reached"] = int(g.stage.max())
        for k in range(4):
            row[f"n_stage{k}"] = int((g.stage == k).sum())
        if len(g) < MIN_EVENTS:
            row.update(rho=np.nan, rho_lo=np.nan, rho_hi=np.nan, verdict="자료 부족")
        else:
            rho, lo, hi = rho_boot(g.total_mm.to_numpy(), g.peak_mm.to_numpy(), rng)
            row.update(rho=rho, rho_lo=lo, rho_hi=hi, verdict="강우와 함께 움직임" if lo > 0 else "확인 안 됨")
        rows.append(row)
    return pd.DataFrame(rows)


def antecedent_check(d: pd.DataFrame) -> tuple[float, float]:
    """보조: 이벤트별 센서 중앙 첨두를 총강우 순위로 회귀한 잔차와 선행 24시간 강우의 순위상관."""
    e = d.groupby("event_id").agg(total=("total_mm", "first"), ant=("antecedent_24h_mm", "first"), med=("peak_mm", "median"))
    rt, rm = rankdata(e.total), rankdata(e.med)
    slope, icpt = np.polyfit(rt, rm, 1)
    resid = rm - (slope * rt + icpt)
    rho, p = spearmanr(e.ant, resid)
    return float(rho), float(p)


def report(s: pd.DataFrame, ant: tuple[float, float], d: pd.DataFrame) -> list[str]:
    cnt = s.verdict.value_counts()
    L = ["[강남 2026] 센서별 강우-상승 반응 (기술 통계, 규칙은 실행 전 고정: src/analysis/rain_response.py)", "=" * 74,
         f"사건 13개 중 센서마다 유효한 사건만 사용. 센서 {len(s)}개, (사건, 센서) {len(d)}개. 부트스트랩 {N_BOOT}회, 시드 {BOOT_SEED}.", "",
         "[분류] " + ", ".join(f"{k} {v}개" for k, v in cnt.items()), "",
         "  센서     n   rho [95% 구간]        분류            30mm미만 첨두 중앙(범위)   50mm이상 첨두 중앙(범위)   최고 단계"]
    f = lambda m, a, b, n: f"{m:6.0f} ({a:.0f}~{b:.0f}, n={n})" if n else "   -"
    for r in s.sort_values("rho", ascending=False, na_position="last").itertuples():
        rho = f"{r.rho:+.2f} [{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]" if np.isfinite(r.rho) else "        -"
        L.append(f"  {r.node_id}  {r.n_events:>2}  {rho:<20}  {r.verdict:<12}  {f(r.light_median_mm, r.light_min_mm, r.light_max_mm, r.light_n):<26} "
                 f"{f(r.heavy_median_mm, r.heavy_min_mm, r.heavy_max_mm, r.heavy_n):<26} {STAGE_NAMES[r.max_stage_reached]}")
    L += ["", f"[보조: 선행 24시간 강우] 총강우로 설명되지 않는 순위 잔차와의 순위상관 {ant[0]:+.2f} (p={ant[1]:.2f}, 사건 13개). 판정 아님.",
          "", "※ 사건 13개라 구간이 넓다. '확인 안 됨'은 반응이 없다는 뜻이 아니라 이 자료로는 구분할 수 없다는 뜻이다.",
          "※ 평소 대비 상승이며 위험이 아니다(관경·침수 이력 없음). 예측 성능을 주장하지 않는다."]
    return L


def main() -> None:
    from src.dashboard.data import load
    D = load()
    info = D["info"]
    d = pairs(D["bt"], D["events"], D["rise"], D["rain_gn"], D["nodes"], info.unstable.to_numpy())
    s = by_sensor(d)
    s.merge(info[["node_id", "role"]], on="node_id").to_csv(OUT_SENSOR, index=False, encoding="utf-8-sig", float_format="%.3f")
    d.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig", float_format="%.3f")
    L = report(s, antecedent_check(d), d)
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
