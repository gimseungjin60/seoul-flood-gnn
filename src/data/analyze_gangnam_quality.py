"""2025-08 강남(구분코드 23) 노드별 수위 데이터 품질(커버리지) 검증 + 6~8월 비교 (원본은 읽기만 한다).

실행: python -m src.data.analyze_gangnam_quality
출력: data/processed/gangnam_202508_quality.csv, gangnam_202508_quality_report.txt

- 위치정보는 2025 원본 파일의 위치정보 컬럼만 쓴다(현재 2026 현황 CSV는 쓰지 않음).
- 측정 간격은 가정하지 않고 노드별 실제 시각 차이의 최빈값으로 정한다.
- 예상 측정 횟수 = 분석 기간 길이 / 노드별 측정 간격, 커버리지 = 고유 측정시각 수 / 예상 측정 횟수.
"""
from pathlib import Path

import pandas as pd

from src.data.analyze_location_match import EN_TO_KO, LOC_COL, PERIOD_CSVS
from src.data.clean_nodes import CODE_COL, ID_COL, NAME_COL, PROCESSED, TIME_COL, normalize_text
from src.data.inspect_water_level import read_chunks

DISTRICT_NAME, DISTRICT_CODE = "강남", "23"
LEVEL_COL, STATUS_COL = "측정수위", "통신상태"
AUG = (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01"))
JUN_AUG = (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-09-01"))
OUT_CSV = PROCESSED / "gangnam_202508_quality.csv"
OUT_REPORT = PROCESSED / "gangnam_202508_quality_report.txt"


def grade(ratio: float) -> str:
    return "HIGH" if ratio >= 0.8 else "MEDIUM" if ratio >= 0.5 else "LOW"


def load_district_rows(path: Path) -> pd.DataFrame:
    """대용량 CSV에서 강남/23 행만 남긴다. 결측/공백은 normalize_text 후 NaN 여부로 판정."""
    parts = []
    for chunk in read_chunks(path):
        chunk = chunk.rename(columns=EN_TO_KO)
        g = chunk[(chunk[NAME_COL] == DISTRICT_NAME) & (chunk[CODE_COL] == DISTRICT_CODE)]
        g = g.assign(**{c: g[c].map(normalize_text) for c in g.columns})
        parts.append(g)
        print(f"  {path.name}: 강남 {sum(map(len, parts)):,}행", end="\r")
    print()
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def node_quality(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """노드별 시작/종료, 측정일수, 행 수, 간격 최빈값, 예상/실제 횟수, 커버리지, 결측 수."""
    df = df[(df["time"] >= start) & (df["time"] < end)]
    period_s = (end - start).total_seconds()
    rows = []
    for node, g in df.groupby(ID_COL):
        t = g["time"].dropna().drop_duplicates().sort_values()
        gaps = t.diff().dt.total_seconds().dropna()
        gaps = gaps[gaps > 0]
        interval = float(gaps.mode().min()) if len(gaps) else float("nan")
        expected = period_s / interval if interval == interval else float("nan")
        locs = g[LOC_COL].dropna().unique() if LOC_COL in g else []
        rows.append({
            "node_id": node,
            "district_code": g[CODE_COL].iloc[0],
            "district_name": g[NAME_COL].iloc[0],
            "location_info": locs[0] if len(locs) else None,
            "start_time": t.min(),
            "end_time": t.max(),
            "measurement_count": len(t),
            "measurement_interval": f"{interval:g}s",
            "coverage_ratio": round(len(t) / expected, 4) if expected == expected else 0.0,
            # --- 아래는 요청 컬럼 외 근거 컬럼 ---
            "n_rows": len(g),
            "n_days": t.dt.date.nunique(),
            "expected_count": round(expected) if expected == expected else None,
            "interval_median_s": float(gaps.median()) if len(gaps) else None,
            "max_gap_hours": round(gaps.max() / 3600, 2) if len(gaps) else None,
            "n_missing_or_blank": int(g[[TIME_COL, LEVEL_COL, STATUS_COL]].isna().sum().sum()),
            "n_bad_time": int(g["time"].isna().sum()),
            "n_dup_time": len(g) - len(t),
            "n_status_not_ok": int((g[STATUS_COL] != "통신양호").sum()),
            "n_location_variants": len(locs),
        })
    q = pd.DataFrame(rows)
    q.insert(q.columns.get_loc("coverage_ratio") + 1, "quality_grade", q["coverage_ratio"].map(grade))
    return q


def grade_summary(q: pd.DataFrame) -> str:
    c = q["quality_grade"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    return (f"노드 {len(q)} | HIGH {c['HIGH']} / MEDIUM {c['MEDIUM']} / LOW {c['LOW']} | "
            f"커버리지 평균 {q['coverage_ratio'].mean():.1%}, 중앙값 {q['coverage_ratio'].median():.1%}")


def daily_availability(rows: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """날짜별로 측정값이 있는 노드 수를 등급별로 센다."""
    g = rows.dropna(subset=["time"]).assign(date=lambda d: d["time"].dt.date)
    g = g.merge(q[["node_id", "quality_grade"]], left_on=ID_COL, right_on="node_id")
    return g.groupby(["date", "quality_grade"])[ID_COL].nunique().unstack(fill_value=0)


def build_report(aug: pd.DataFrame, full: pd.DataFrame, monthly: dict[str, pd.DataFrame],
                 daily: pd.DataFrame) -> str:
    L = []
    w = L.append
    cols = ["node_id", "coverage_ratio", "measurement_count", "expected_count", "start_time", "end_time",
            "max_gap_hours"]
    w("[강남(23)] 2025-08 수위 데이터 품질 리포트")
    w("=" * 60)
    w(f"대상 파일: data/raw/{PERIOD_CSVS[2].name} (위치정보: 같은 파일의 위치정보 컬럼)")
    w(f"분석 기간: {AUG[0]} ~ {AUG[1]} (미포함)")
    w("\n1. 측정 간격 (노드별 실제 시각 차이 최빈값, 가정 없음)")
    w("  " + ", ".join(f"{k}: {v}개 노드" for k, v in aug["measurement_interval"].value_counts().items()))
    w("  예상 측정 횟수 = 기간 길이(초) / 노드별 간격(초), 커버리지 = 고유 측정시각 수 / 예상 횟수")
    w("\n2. 8월 강남 요약")
    w(f"  {grade_summary(aug)}")
    w(f"  결측/공백 값 합계 {aug['n_missing_or_blank'].sum()}, 시각 파싱 실패 {aug['n_bad_time'].sum()}, "
      f"중복 측정시각 {aug['n_dup_time'].sum()}, 통신양호 외 상태 {aug['n_status_not_ok'].sum()}")
    w(f"  위치정보 2종 이상인 노드 {int((aug['n_location_variants'] > 1).sum())}, "
      f"위치정보 없는 노드 {int(aug['location_info'].isna().sum())}")
    w(f"  측정 시작일 분포: " + ", ".join(
        f"{d}: {n}" for d, n in aug["start_time"].dt.date.value_counts().sort_index().items()))
    w("\n3. 가장 데이터가 부족한 노드 10개")
    w(aug.nsmallest(10, "coverage_ratio")[cols].to_string(index=False))
    w("\n4. 가장 데이터가 충분한 노드 10개")
    w(aug.nlargest(10, "coverage_ratio")[cols].to_string(index=False))
    w("\n5. 8월을 연구기간으로 사용할 경우 실제 사용 가능한 노드 수")
    hi = aug["quality_grade"] == "HIGH"
    w(f"  HIGH(>=80%): {hi.sum()} / HIGH+MEDIUM(>=50%): {(aug['quality_grade'] != 'LOW').sum()} / 전체 {len(aug)}")

    w("\n   날짜별 측정값이 있는 노드 수 (등급별)")
    w(daily.to_string())
    w("\n6. 6~8월 전체 vs 8월 비교 (같은 커버리지 정의, 기간만 다름)")
    w(f"  [8월]    {grade_summary(aug)}")
    w(f"  [6~8월]  {grade_summary(full)}")
    for m, q in monthly.items():
        w(f"  [{m}] {grade_summary(q)}")
    allm = set.intersection(*(set(q.loc[q['quality_grade'] == 'HIGH', 'node_id']) for q in monthly.values()))
    w(f"  6·7·8월 모두 HIGH인 노드: {len(allm)}")
    tot_aug = aug["measurement_count"].sum()
    tot_full = full["measurement_count"].sum()
    w(f"  전체 측정 건수: 8월 {tot_aug:,} / 6~8월 {tot_full:,}")
    w(f"  HIGH 노드 × 기간 내 측정시각: 8월 {aug.loc[hi, 'measurement_count'].sum():,} / "
      f"6~8월 {full.loc[full['quality_grade'] == 'HIGH', 'measurement_count'].sum():,}")
    w("\n  6~8월 노드별 커버리지 상위/하위")
    w(full.sort_values("coverage_ratio")[["node_id", "coverage_ratio", "start_time", "end_time"]]
      .to_string(index=False, max_rows=20))
    return "\n".join(L) + "\n"


def main() -> None:
    frames = {p.stem[-6:]: load_district_rows(p) for p in PERIOD_CSVS}
    aug = node_quality(frames["202508"], *AUG)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    aug.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    all_rows = pd.concat(frames.values(), ignore_index=True)
    full = node_quality(all_rows, *JUN_AUG)
    month_bounds = {"202506": ("2025-06-01", "2025-07-01"), "202507": ("2025-07-01", "2025-08-01"),
                    "202508": AUG}
    monthly = {m: node_quality(frames[m], pd.Timestamp(a), pd.Timestamp(b))
               for m, (a, b) in month_bounds.items()}
    report = build_report(aug, full, monthly, daily_availability(frames["202508"], aug))
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
