"""강우 강도별 과거 상승 단계 요약(대시보드 '참고' 근거). 판정이 아니라 기술 통계다. 모델 학습 없음. 결과를 보고 규칙을 바꾸지 않는다.

실행: python -m src.analysis.rain_vs_rise
출력: data/processed/rain_vs_rise_by_event.csv, rain_vs_rise_report.txt

- 대상: 2026-06~08 평가 이벤트 13개, 기록이 안정적인 센서(평가 대상 + 불안정이 아닌 이웃, 사건마다 38~39개).
- 사건별 센서 첨두 상승 = 사건 시작 ~ 종료+3시간 창 안 센서별 최대 상승폭(평소 대비, 지도와 같은 정의, docs/16).
- 강우량 = 사건 총강우량(mm)과 10분 최대 강우(mm). 값은 이벤트 정의(prepare_rainfall_2026)의 것을 그대로 쓴다.
- 이벤트가 13개뿐이라 구간이 넓다. 인과나 예측 성능이 아니라 '과거에 이런 비에서 이랬다'는 요약이다.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.rise_levels import STAGE_NAMES, stage
from src.data.clean_nodes import PROCESSED

OUT_CSV = PROCESSED / "rain_vs_rise_by_event.csv"
OUT_REPORT = PROCESSED / "rain_vs_rise_report.txt"
TAIL = pd.Timedelta(hours=3)
BANDS = ((0, 30, "30 mm 미만"), (30, 50, "30~50 mm"), (50, 1e9, "50 mm 이상"))


def by_event(bt, events: pd.DataFrame, rise: np.ndarray, unstable: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        w = np.flatnonzero((bt >= r.start) & (bt <= r.end + TAIL))
        x = np.where(np.isfinite(rise[w]), rise[w], -np.inf)
        pk = x.max(axis=0)
        ok = ~unstable & np.isfinite(pk) & (pk > -1e9)
        p, s = pk[ok], stage(pk[ok])
        rows.append(dict(event_id=r.event_id, start=r.start, total_mm=r.total_mm, max_10min_mm=r.max_10min_mm, duration_h=r.duration_h,
                         n_sensors=int(ok.sum()), median_peak_mm=float(np.median(p)), p90_peak_mm=float(np.percentile(p, 90)),
                         n_up=int((s >= 1).sum()), n_big=int((s >= 2).sum()), n_vbig=int((s >= 3).sum())))
    return pd.DataFrame(rows)


def report(d: pd.DataFrame) -> list[str]:
    L = ["[강남 2026] 강우 강도별 과거 상승 단계 요약 (기술 통계, docs/16 단계 기준)", "=" * 70,
         f"이벤트 {len(d)}개, 센서 {int(d.n_sensors.min())}~{int(d.n_sensors.max())}개(기록 안정). 첨두 = 사건 창+3시간 안 센서별 최대 상승폭.", "",
         "[이벤트별]", "  번호  시작    총강우  10분최대  센서중앙첨두  '상승'이상  '큰'이상  '매우 큰'"]
    for r in d.itertuples():
        L.append(f"  {r.event_id:>3}  {r.start:%m-%d}  {r.total_mm:6.1f}  {r.max_10min_mm:7.1f}  {r.median_peak_mm:9.0f}mm  "
                 f"{r.n_up:>4}/{r.n_sensors}  {r.n_big:>4}/{r.n_sensors}  {r.n_vbig:>4}/{r.n_sensors}")
    L += ["", "[총강우량 구간별]", "  구간          이벤트  센서 중앙 첨두(mm)  '큰 상승' 이상 센서 수 (최소~최대)  '매우 큰 상승' 센서 수 (최소~최대)"]
    for lo, hi, name in BANDS:
        g = d[(d.total_mm >= lo) & (d.total_mm < hi)]
        if len(g):
            L.append(f"  {name:<10}  {len(g):>4}   {g.median_peak_mm.min():.0f}~{g.median_peak_mm.max():.0f}"
                     f"            {g.n_big.min()}~{g.n_big.max()}                          {g.n_vbig.min()}~{g.n_vbig.max()}")
    L += ["", "[순위상관(스피어먼)]"]
    for c, cn in (("total_mm", "총강우량"), ("max_10min_mm", "10분 최대 강우")):
        for m, mn in (("median_peak_mm", "센서 중앙 첨두"), ("n_vbig", "'매우 큰 상승' 센서 수")):
            rho, p = spearmanr(d[c], d[m])
            L.append(f"  {cn} vs {mn}: rho={rho:.2f} (p={p:.3f})")
    L += ["", "※ 이벤트 13개라 구간이 넓다. 같은 강우라도 사건마다 편차가 크다(예: 총 16 mm인 08-16에 '매우 큰 상승' 6개).",
          "※ '위험'이 아니라 평소 대비 상승이다. 관경·침수 이력이 없어 위험 기준은 정할 수 없다."]
    return L


def main() -> None:
    from src.dashboard.data import load
    D = load()
    d = by_event(D["bt"], D["events"], D["rise"], D["info"].unstable.to_numpy())
    d.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.3f")
    L = report(d)
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
