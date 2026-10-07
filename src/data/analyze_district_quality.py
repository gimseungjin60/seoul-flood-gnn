"""서울시 전 자치구 2025-06~08 수위 노드 품질(커버리지) 비교 (원본은 읽기만 한다).

실행: python -m src.data.analyze_district_quality
출력: data/processed/district_quality_comparison.csv, district_quality_report.txt
      (+ 근거용 노드 단위 표 data/processed/district_quality_nodes.csv)

- 커버리지 정의는 강남 분석(analyze_gangnam_quality)과 같다:
  노드별 실제 측정시각 차이 최빈값 = 간격, 예상 횟수 = 기간 길이 / 간격, 커버리지 = 고유 측정시각 수 / 예상 횟수.
- 기간: 2025-06-01 00:00 ~ 2025-09-01 00:00 (미포함). HIGH >= 80%, MEDIUM 50~80%, LOW < 50%.
- 위치정보는 2025 원본 파일의 위치정보 컬럼만 쓴다. 주소/좌표는 만들지 않는다.
- 공간 분산은 주소 문자열의 도로명/법정동 이름 개수로만 정성 요약한다.
"""
import re

import numpy as np
import pandas as pd

from src.data.analyze_gangnam_quality import JUN_AUG, AUG, grade
from src.data.analyze_location_match import EN_TO_KO, LOC_COL, PERIOD_CSVS
from src.data.clean_nodes import CODE_COL, ID_COL, NAME_COL, PROCESSED, TIME_COL, normalize_text
from src.data.inspect_water_level import read_chunks

OUT_CSV = PROCESSED / "district_quality_comparison.csv"
OUT_NODES = PROCESSED / "district_quality_nodes.csv"
OUT_REPORT = PROCESSED / "district_quality_report.txt"
FOCUS = ["강남", "서초", "동작"]
TIME_FMT = "%Y-%m-%d %H:%M:%S"
_ROAD = re.compile(r"([가-힣]+(?:대로|로))(?:\s*\d+\s*번?길|\s*\d|\s|$|앞|\()")
_DONG = re.compile(r"([가-힣]+\d*동)(?:\s*\d|\s|$)")


def node_coverage(ids: np.ndarray, secs: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """ids(int), secs(epoch 초) → 노드별 고유 측정시각 수, 간격 최빈값/중앙값, 시작/종료, 커버리지."""
    lo, hi = start.value // 10**9, end.value // 10**9
    keep = (secs >= lo) & (secs < hi)
    df = pd.DataFrame({"nid": ids[keep], "t": secs[keep]}).drop_duplicates().sort_values(["nid", "t"])
    gap = df.groupby("nid")["t"].diff()
    gaps = pd.DataFrame({"nid": df["nid"], "gap": gap})[gap > 0]
    mode = gaps.groupby("nid")["gap"].agg(lambda s: s.mode().min())
    out = df.groupby("nid")["t"].agg(measurement_count="size", start="min", end="max")
    out["interval_mode_s"] = mode
    out["interval_median_s"] = gaps.groupby("nid")["gap"].median()
    expected = (hi - lo) / out["interval_mode_s"]
    out["coverage_ratio"] = (out["measurement_count"] / expected).fillna(0.0).round(4)
    for c in ("start", "end"):
        out[c] = pd.to_datetime(out[c], unit="s")
    return out


def scan(paths=PERIOD_CSVS):
    """3개월 파일 → (nid 배열, epoch초 배열, 노드 메타(구분코드/구분명/위치정보/월별 존재), 시각 파싱 실패 수)."""
    id_index: dict[str, int] = {}
    ids, secs, metas, bad_time = [], [], [], 0
    for p in paths:
        month = p.stem[-6:]
        for chunk in read_chunks(p):
            chunk = chunk.rename(columns=EN_TO_KO)
            codes, uniq = pd.factorize(chunk[ID_COL])
            gmap = np.array([id_index.setdefault(u, len(id_index)) for u in uniq], dtype=np.int32)
            t = pd.to_datetime(chunk[TIME_COL], format=TIME_FMT, errors="coerce")
            bad_time += int(t.isna().sum())
            ok = t.notna().to_numpy()
            ids.append(gmap[codes][ok])
            secs.append(t[ok].to_numpy().astype("datetime64[s]").astype(np.int64))  # pandas 3은 us 단위
            m = chunk.groupby([ID_COL, CODE_COL, NAME_COL, LOC_COL], dropna=False).size().reset_index(name="n")
            metas.append(m.assign(month=month))
            print(f"  {p.name}: 노드 {len(id_index)}", end="\r")
        print()
    meta = pd.concat(metas, ignore_index=True)
    meta["nid"] = meta[ID_COL].map(id_index)
    return np.concatenate(ids), np.concatenate(secs), meta, bad_time


def node_table(meta: pd.DataFrame, full: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    meta = meta.assign(loc=meta[LOC_COL].map(normalize_text))
    months = meta.pivot_table(index="nid", columns="month", values="n", aggfunc="sum").notna()
    base = meta.groupby("nid").agg(
        node_id=(ID_COL, "first"), district_code=(CODE_COL, "first"), district_name=(NAME_COL, "first"),
        n_names=(NAME_COL, "nunique"), n_loc_variants=("loc", "nunique"),
        location_info=("loc", lambda s: s.dropna().iloc[0] if s.notna().any() else None))
    base = base.join(months.add_prefix("in_")).join(full)
    base["quality_grade"] = base["coverage_ratio"].map(grade)
    base["aug_coverage"] = aug["coverage_ratio"].reindex(base.index).fillna(0.0)
    return base.reset_index(drop=True).sort_values("node_id")


def spatial_summary(locs: pd.Series) -> dict:
    """주소 문자열에서 도로명(…로/대로)·법정동(…동) 이름만 뽑아 분산 정도를 센다. 좌표 아님."""
    roads = locs.dropna().map(lambda s: (_ROAD.findall(s) or [None])[0])
    dongs = locs.dropna().map(lambda s: (_DONG.findall(s) or [None])[0])
    keys = roads.fillna(dongs)  # 도로명 우선, 없으면 동
    vc = keys.dropna().value_counts()
    return {
        "n_roads": int(roads.dropna().nunique()), "n_dongs": int(dongs.dropna().nunique()),
        "n_area_keys": int(len(vc)), "top_key": vc.index[0] if len(vc) else None,
        "top_share": float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan"),
        "n_unparsed": int(keys.isna().sum()),
        "top3": ", ".join(f"{k}({v})" for k, v in vc.head(3).items()),
    }


def district_table(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (code, name), g in nodes.groupby(["district_code", "district_name"]):
        stable = g[g["coverage_ratio"] >= 0.5]
        high = g[g["quality_grade"] == "HIGH"]
        sp = spatial_summary(stable["location_info"])
        rows.append({
            "district_name": name, "district_code": code, "total_nodes": len(g),
            "high_nodes": len(high), "medium_nodes": int((g["quality_grade"] == "MEDIUM").sum()),
            "nodes_50plus": len(stable), "located_stable_nodes": int(stable["location_info"].notna().sum()),
            "mean_coverage": round(stable["coverage_ratio"].mean(), 4) if len(stable) else None,
            "median_coverage": round(stable["coverage_ratio"].median(), 4) if len(stable) else None,
            # --- 아래는 요청 컬럼 외 근거 컬럼 ---
            "jun_nodes": int(g["in_202506"].sum()), "jul_nodes": int(g["in_202507"].sum()),
            "aug_nodes": int(g["in_202508"].sum()),
            "stable_ratio": round(len(stable) / len(g), 4),
            "all_nodes_mean_coverage": round(g["coverage_ratio"].mean(), 4),
            "aug_high_nodes": int((g["aug_coverage"] >= 0.8).sum()),
            "stable_start": stable["start"].min(), "stable_end": stable["end"].max(),
            "stable_interval_mode_s": stable["interval_mode_s"].mode().min() if len(stable) else None,
            "stable_interval_median_s": stable["interval_median_s"].median() if len(stable) else None,
            "meets_20_stable": len(stable) >= 20,
            "stable_roads": sp["n_roads"], "stable_dongs": sp["n_dongs"],
            "stable_area_keys": sp["n_area_keys"], "stable_top_area_share": round(sp["top_share"], 3),
            "stable_top_areas": sp["top3"], "stable_unparsed_locations": sp["n_unparsed"],
        })
    t = pd.DataFrame(rows)
    # 1순위 HIGH, 2순위 50%+, 3순위 위치정보 보유 안정노드, 4순위(20개 이상)는 표시만
    t = t.sort_values(["high_nodes", "nodes_50plus", "located_stable_nodes"], ascending=False)
    t["rank"] = range(1, len(t) + 1)
    return t.reset_index(drop=True)


def spread_label(r) -> str:
    if r["nodes_50plus"] == 0:
        return "안정 노드 없음"
    if r["nodes_50plus"] < 5:
        return "노드 수가 적어 판단 불가"
    share = r["stable_top_area_share"]
    return "한 도로/동에 몰림" if share >= 0.4 else "다소 몰림" if share >= 0.25 else "여러 도로/동에 분산"


def build_report(t: pd.DataFrame, nodes: pd.DataFrame, bad_time: int) -> str:
    L = []
    w = L.append
    w("서울시 자치구별 2025-06~08 하수관로 수위 노드 품질 비교")
    w("=" * 70)
    w("대상 파일: " + ", ".join(f"data/raw/{p.name}" for p in PERIOD_CSVS))
    w(f"기간: {JUN_AUG[0]} ~ {JUN_AUG[1]} (미포함), 위치정보 출처: 2025 원본 파일의 위치정보 컬럼")
    w("커버리지 = 고유 측정시각 수 / (기간 길이 / 노드별 실제 간격 최빈값). 안정 노드 = 커버리지 50% 이상")
    w(f"전체 노드 {len(nodes)}, 시각 파싱 실패 행 {bad_time}, 구분명 2종 이상 노드 {int((nodes['n_names'] > 1).sum())}, "
      f"위치정보 2종 이상 노드 {int((nodes['n_loc_variants'] > 1).sum())}, 위치정보 없는 노드 {int(nodes['location_info'].isna().sum())}")
    w("노드별 간격 최빈값 분포: " + ", ".join(
        f"{k:g}s: {v}" for k, v in nodes["interval_mode_s"].value_counts().items()))

    w("\n1. 전체 자치구 순위 (1순위 HIGH ↓, 2순위 50%+ ↓, 3순위 위치정보 보유 안정노드 ↓; * = 강남/서초/동작)")
    w(f"{'순위':>3} {'자치구':<6} {'전체':>4} {'6월':>4} {'7월':>4} {'8월':>4} {'HIGH':>5} {'MED':>4} {'50%+':>5} "
      f"{'위치보유':>5} {'평균cov':>8} {'중앙cov':>8} {'8월HIGH':>7} {'20+':>4}  공간 분산(주소 기준)")
    for _, r in t.iterrows():
        mark = "*" if r["district_name"] in FOCUS else " "
        mc = f"{r['mean_coverage']:.1%}" if pd.notna(r["mean_coverage"]) else "-"
        md = f"{r['median_coverage']:.1%}" if pd.notna(r["median_coverage"]) else "-"
        w(f"{r['rank']:>3}{mark}{r['district_name']:<6} {r['total_nodes']:>4} {r['jun_nodes']:>4} {r['jul_nodes']:>4} "
          f"{r['aug_nodes']:>4} {r['high_nodes']:>5} {r['medium_nodes']:>4} {r['nodes_50plus']:>5} "
          f"{r['located_stable_nodes']:>5} {mc:>8} {md:>8} {r['aug_high_nodes']:>7} "
          f"{'O' if r['meets_20_stable'] else 'X':>4}  {spread_label(r)}")
    w(f"\n  안정 노드(50%+) 20개 이상 자치구: {int(t['meets_20_stable'].sum())}개")

    def detail(r):
        w(f"\n[{r['rank']}위] {r['district_name']} (코드 {r['district_code']})")
        w(f"  노드: 전체 {r['total_nodes']} (6월 {r['jun_nodes']} / 7월 {r['jul_nodes']} / 8월 {r['aug_nodes']})")
        w(f"  HIGH {r['high_nodes']} / MEDIUM {r['medium_nodes']} / 50%+ {r['nodes_50plus']} "
          f"(안정 비율 {r['stable_ratio']:.1%}), 위치정보 보유 안정노드 {r['located_stable_nodes']}")
        if r["nodes_50plus"]:
            w(f"  안정노드 커버리지 평균 {r['mean_coverage']:.1%}, 중앙값 {r['median_coverage']:.1%} / "
              f"측정기간 {r['stable_start']} ~ {r['stable_end']} / 간격 최빈값 {r['stable_interval_mode_s']:g}s, "
              f"중앙값 {r['stable_interval_median_s']:g}s")
            w(f"  공간 분산: {spread_label(r)} — 도로명 {r['stable_roads']}종, 법정동 {r['stable_dongs']}종, "
              f"최다 지역 비중 {r['stable_top_area_share']:.0%}, 상위: {r['stable_top_areas']}, "
              f"도로/동 추출 불가 {r['stable_unparsed_locations']}")
        w(f"  8월만 쓸 때 HIGH: {r['aug_high_nodes']} (참고)")
        g = nodes[(nodes["district_name"] == r["district_name"]) & (nodes["coverage_ratio"] >= 0.5)]
        for _, n in g.head(8).iterrows():
            w(f"    {n['node_id']} {n['coverage_ratio']:.1%} {n['location_info']}")
        if len(g) > 8:
            w(f"    ... 외 {len(g) - 8}개 (district_quality_nodes.csv 참고)")

    w("\n2. TOP 5 상세")
    for _, r in t.head(5).iterrows():
        detail(r)
    w("\n3. 강남 / 서초 / 동작")
    for name in FOCUS:
        sel = t[t["district_name"] == name]
        if len(sel):
            detail(sel.iloc[0])
        else:
            w(f"\n{name}: 데이터 없음")

    w("\n4. 요약 지표")
    best_cov = t[t["nodes_50plus"] >= 5].sort_values("mean_coverage", ascending=False).head(1)
    w(f"  HIGH 노드 최다: {t.iloc[0]['district_name']} ({t.iloc[0]['high_nodes']})")
    w(f"  전체 노드 최다: {t.sort_values('total_nodes').iloc[-1]['district_name']} "
      f"({t['total_nodes'].max()})")
    if len(best_cov):
        b = best_cov.iloc[0]
        w(f"  안정노드 5개 이상 중 평균 커버리지 최고: {b['district_name']} ({b['mean_coverage']:.1%}, 안정노드 {b['nodes_50plus']})")
    w(f"  서울 전체 안정 노드: HIGH {int((nodes['quality_grade'] == 'HIGH').sum())}, "
      f"50%+ {int((nodes['coverage_ratio'] >= 0.5).sum())} / {len(nodes)}")
    return "\n".join(L) + "\n"


def main() -> None:
    ids, secs, meta, bad_time = scan()
    full = node_coverage(ids, secs, *JUN_AUG)
    aug = node_coverage(ids, secs, *AUG)
    del ids, secs
    nodes = node_table(meta, full, aug)
    t = district_table(nodes)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUT_NODES, index=False, encoding="utf-8-sig")
    t.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    report = build_report(t, nodes, bad_time)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
