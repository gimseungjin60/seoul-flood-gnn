"""2025-06~08 수위 시계열 고유번호 ↔ 현재 위치정보 CSV 고유번호 매칭 검증 (원본은 읽기만 한다).

실행: python -m src.data.analyze_location_match [구분명 ...]   (기본: 강남)
출력: data/processed/{slug}_location_match.csv, {slug}_location_match_report.txt
      (강남 → gangnam, 서초 → seocho, 동작 → dongjak, 그 외 → 구분코드)

- 주소/좌표를 추측하거나 수정하지 않는다. 위치정보는 원본 문자열 그대로 쓴다(공백만 정리).
- 2025 파일에도 위치정보 컬럼이 있으면 참고용으로 현재 위치정보와 일치 여부만 비교한다.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_nodes import CODE_COL, ID_COL, NAME_COL, PROCESSED, TIME_COL, normalize_text
from src.data.inspect_water_level import PROJECT_ROOT, read_chunks

RAW = PROJECT_ROOT / "data" / "raw"
PERIOD_CSVS = [RAW / f"tv_swm_wal_mea_2025{m}.csv" for m in ("06", "07", "08")]
LOCATION_CSV = RAW / "서울시 하수관로 수위 현황.csv"
LOC_COL = "위치정보"
# 202507/202508 파일은 영문 헤더(UTF-8), 202506은 한글 헤더(cp949). 컬럼 순서는 동일.
EN_TO_KO = {"unq_no": ID_COL, "se_cd": CODE_COL, "se_nm": NAME_COL, "msrmt_ymd": TIME_COL,
            "msrmt_watl": "측정수위", "sgn_stts": "통신상태", "pstn_info": LOC_COL}
SLUGS = {"강남": "gangnam", "서초": "seocho", "동작": "dongjak"}

# 위치정보 유형 분류(단일 라벨, 위에서부터 우선). 패턴에 맞는지 보기만 하고 값은 바꾸지 않는다.
_ROAD_NO = re.compile(r"[가-힣0-9]+(?:로|길)\s*\d+(?:번?길)?\s*\d*(?:-\d+)?")  # 강남대로 399, 논현로85길 56
_LOT_NO = re.compile(r"[가-힣0-9]+동\s*\d+(?:-\d+)?")                          # 역삼동 820-5
_GU = re.compile(r"(?:서울(?:특별시|시)?\s*)?[가-힣]+구\s")
_LANDMARK = re.compile(r"맨홀|빌딩|아파트|사거리|교차로|역|학교|공원|펌프장|센터|주민센터|앞|옆|뒤|건너|입구|빗물|박스")
LOC_TYPES = ["주소 형태", "도로명/지번 형태", "랜드마크/시설물 설명", "기타/불명확", "빈 값"]


def classify_location(value) -> str:
    """주소 형태 = '○○구' + 도로명/지번 번호, 도로명/지번 형태 = 구 없이 번호만,
    랜드마크 = 번호 없이 시설물 설명만, 나머지 = 기타/불명확."""
    s = normalize_text(value)
    if s is None:
        return "빈 값"
    has_no = bool(_ROAD_NO.search(s) or _LOT_NO.search(s))
    if has_no and _GU.match(s):
        return "주소 형태"
    if has_no:
        return "도로명/지번 형태"
    if _LANDMARK.search(s):
        return "랜드마크/시설물 설명"
    return "기타/불명확"


def scan_period_file(path: Path) -> tuple[dict, pd.DataFrame]:
    """대용량 수위 CSV 1개 → (파일 통계, 고유번호·구분코드·구분명·위치정보 조합별 측정 건수)."""
    n_rows, nulls, hashes, parts = 0, None, [], []
    uniq = {c: set() for c in (ID_COL, CODE_COL, NAME_COL, TIME_COL)}
    t_min = t_max = None
    for chunk in read_chunks(path):
        chunk = chunk.rename(columns=EN_TO_KO)
        n_rows += len(chunk)
        nulls = chunk.isna().sum() if nulls is None else nulls + chunk.isna().sum()
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        for c in uniq:
            uniq[c].update(chunk[c].dropna().unique())
        t = chunk[TIME_COL].dropna()
        if len(t):
            t_min = min(filter(None, [t_min, t.min()]))
            t_max = max(filter(None, [t_max, t.max()]))
        keys = [ID_COL, CODE_COL, NAME_COL] + ([LOC_COL] if LOC_COL in chunk else [])
        parts.append(chunk.groupby(keys, dropna=False).size())
        print(f"  {path.name}: {n_rows:,}행", end="\r")
    print()
    nodes = pd.concat(parts).groupby(level=list(range(len(keys))), dropna=False).sum()
    nodes = nodes.rename("n_records").reset_index()
    stats = {
        "file": path.name, "rows": n_rows,
        "n_id": len(uniq[ID_COL]), "n_code": len(uniq[CODE_COL]), "n_name": len(uniq[NAME_COL]),
        "start": t_min, "end": t_max, "n_timestamps": len(uniq[TIME_COL]),
        "n_days": len({str(x)[:10] for x in uniq[TIME_COL]}),
        "nulls": nulls.to_dict(),
        "dup_rows": n_rows - len(np.unique(np.concatenate(hashes))),
        "names": sorted(uniq[NAME_COL]),
    }
    return stats, nodes


def load_location_master(path: Path = LOCATION_CSV) -> tuple[dict, pd.DataFrame]:
    """현재 현황 CSV → 고유번호당 1행 위치 마스터. 통계는 행이 아니라 고유번호 기준."""
    df = pd.concat(read_chunks(path))
    df[LOC_COL] = df[LOC_COL].map(normalize_text)
    per_id = df.groupby(ID_COL).agg(
        district_code=(CODE_COL, "first"), district_name=(NAME_COL, "first"),
        n_code=(CODE_COL, "nunique"), n_loc=(LOC_COL, "nunique"),
        location_info=(LOC_COL, lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        n_rows=(ID_COL, "size"),
    ).reset_index().rename(columns={ID_COL: "node_id"})
    stats = {
        "file": path.name, "rows": len(df), "n_id": len(per_id),
        "time_range": (df[TIME_COL].min(), df[TIME_COL].max()),
        "ids_by_name": per_id.groupby(["district_code", "district_name"]).size(),
        "n_loc_empty": int(per_id["location_info"].isna().sum()),
        "n_loc_present": int(per_id["location_info"].notna().sum()),
        "n_dup_ids": int((per_id["n_rows"] > 1).sum()),  # 여러 측정시각으로 반복된 고유번호 수
        "n_multi_loc": int((per_id["n_loc"] > 1).sum()),  # 같은 고유번호에 위치정보가 2개 이상
        "n_multi_code": int((per_id["n_code"] > 1).sum()),
    }
    return stats, per_id


def join_status(period_ids: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """period_ids(node_id, district_code, district_name) ⟗ master → match_status.
    MATCHED = 양쪽 존재 + 위치정보 비어있지 않음, NO_LOCATION = 2025에만 있거나 위치정보 빈 값,
    LOCATION_ONLY = 현재 위치정보에만 존재."""
    m = period_ids.merge(master[["node_id", "district_code", "district_name", "location_info"]],
                         on="node_id", how="outer", suffixes=("", "_loc"), indicator=True)
    m["exists_in_2025_06_08"] = m["_merge"] != "right_only"
    for c in ("district_code", "district_name"):
        m[c] = m[c].fillna(m[f"{c}_loc"])
    m["match_status"] = np.select(
        [m["_merge"] == "right_only", (m["_merge"] == "both") & m["location_info"].notna()],
        ["LOCATION_ONLY", "MATCHED"], "NO_LOCATION")
    cols = ["node_id", "district_code", "district_name", "location_info",
            "exists_in_2025_06_08", "match_status"]
    return m[cols].sort_values("node_id").reset_index(drop=True)


def period_id_table(nodes: pd.DataFrame) -> pd.DataFrame:
    """3개월 조합 테이블 → 고유번호당 1행 (구분코드/구분명은 최빈값 대신 첫 값, 혼재는 리포트에서 따로 셈)."""
    t = nodes.assign(node_id=nodes[ID_COL].map(normalize_text))
    return (t.groupby("node_id")
            .agg(district_code=(CODE_COL, "first"), district_name=(NAME_COL, "first"),
                 n_records=("n_records", "sum"), n_name=(NAME_COL, "nunique"))
            .reset_index())


def build_report(district: str, period_stats, all_nodes, per_month, master_stats, master,
                 joined_all, result, loc2025: pd.DataFrame | None = None) -> str:
    L = []
    w = L.append
    gu = result
    n_2025 = int(gu["exists_in_2025_06_08"].sum())
    matched = gu[gu["match_status"] == "MATCHED"]
    no_loc = gu[gu["match_status"] == "NO_LOCATION"]
    loc_only = gu[gu["match_status"] == "LOCATION_ONLY"]
    n_master_gu = int((master["district_name"] == district).sum())
    rate = len(matched) / n_2025 if n_2025 else float("nan")

    w(f"[{district}] 위치정보 매칭 검증 리포트")
    w("=" * 60)
    w("\n1. 분석 대상 파일")
    for s in period_stats:
        w(f"  - data/raw/{s['file']}")
    w(f"  - data/raw/{master_stats['file']} (위치 마스터)")
    w("\n2. 분석 기간")
    w(f"  {min(s['start'] for s in period_stats)} ~ {max(s['end'] for s in period_stats)}")

    w("\n3. 2025-06~08 파일별 통계")
    for s in period_stats:
        w(f"  [{s['file']}]")
        w(f"    행 {s['rows']:,} / 고유번호 {s['n_id']:,} / 구분코드 {s['n_code']} / 구분명 {s['n_name']}")
        w(f"    기간 {s['start']} ~ {s['end']} / 측정일자(시각) {s['n_timestamps']:,}개, 날짜 {s['n_days']}일")
        w(f"    결측 {s['nulls']} / 완전중복 행 {s['dup_rows']:,}")
    w(f"  구분명 실제 값: {period_stats[0]['names']}")

    w("\n4. 전체 노드 수")
    w(f"  2025-06~08 합집합 고유번호: {len(all_nodes):,}")
    w(f"  현재 위치정보 CSV 고유번호: {master_stats['n_id']:,} (행 {master_stats['rows']:,}, "
      f"측정시각 {master_stats['time_range'][0]} ~ {master_stats['time_range'][1]})")
    w(f"    위치정보 있음 {master_stats['n_loc_present']} / 빈 값 {master_stats['n_loc_empty']} / "
      f"여러 행 반복 고유번호 {master_stats['n_dup_ids']} / 위치정보 2종 이상 {master_stats['n_multi_loc']} / "
      f"구분코드 2종 이상 {master_stats['n_multi_code']}")
    vc = joined_all["match_status"].value_counts()
    w(f"  전체 JOIN: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
    w("  현재 위치정보 구분명별 고유번호 수:")
    w("    " + ", ".join(f"{c}{n}:{v}" for (c, n), v in master_stats["ids_by_name"].items()))

    w(f"\n5. {district} 노드 수")
    w(f"  2025-06~08 {district} 고유번호: {n_2025}")
    for m, ids in per_month.items():
        w(f"    {m}: {len(ids)}")
    w(f"  현재 위치정보 CSV {district} 고유번호: {n_master_gu}")
    w(f"  MATCHED: {len(matched)} / NO_LOCATION: {len(no_loc)} / LOCATION_ONLY: {len(loc_only)}")
    w(f"\n6. 매칭률 (MATCHED / 2025 {district} 고유번호): {len(matched)}/{n_2025} = {rate:.1%}")

    w("\n7. 위치정보 없는 노드 목록 (NO_LOCATION)")
    w("  " + (", ".join(no_loc["node_id"]) or "없음"))
    w("  현재 위치정보에만 있는 노드 (LOCATION_ONLY)")
    w("  " + (", ".join(loc_only["node_id"]) or "없음"))
    month_cols = list(per_month)
    partial = [i for i in gu.loc[gu["exists_in_2025_06_08"], "node_id"]
               if sum(i in per_month[m] for m in month_cols) < len(month_cols)]
    w(f"  3개월 중 일부 월에만 존재하는 노드: {', '.join(partial) or '없음'}")

    types = matched["location_info"].map(classify_location).value_counts().reindex(LOC_TYPES, fill_value=0)
    w("\n8. MATCHED 위치정보 유형 (자동 패턴 분류, 값은 수정하지 않음)")
    for k, v in types.items():
        w(f"  {k}: {v}")
    w("  위치정보 샘플 (전체)")
    for _, r in matched.iterrows():
        w(f"    {r['node_id']} [{classify_location(r['location_info'])}] {r['location_info']}")

    w("\n9. 데이터 품질 문제")
    html = matched[matched["location_info"].str.contains(r"&lt;|&gt;|&amp;", na=False)]
    w(f"  - HTML 엔티티(&lt; 등) 포함 위치정보: {', '.join(html['node_id']) or '없음'}")
    w(f"  - 구분명 2종 이상 섞인 2025 고유번호: "
      f"{', '.join(all_nodes.loc[all_nodes['n_name'] > 1, 'node_id']) or '없음'}")
    if loc2025 is not None:
        cmp = matched.merge(loc2025, on="node_id", how="left")
        diff = cmp[cmp["loc_2025"] != cmp["location_info"]]
        w(f"  - 2025 파일에도 위치정보 컬럼 존재. MATCHED 중 2025 위치정보와 현재 위치정보가 다른 노드: "
          f"{len(diff)}")
        for _, r in diff.iterrows():
            w(f"      {r['node_id']}: 2025='{r['loc_2025']}' / 현재='{r['location_info']}'")
        nl = no_loc.merge(loc2025, on="node_id", how="left")
        has = nl[nl["loc_2025"].notna()]
        w(f"  - NO_LOCATION {len(nl)}개 중 2025 파일 자체 위치정보가 있는 노드: {len(has)} "
          f"(현재 위치 마스터 기준 match_status는 그대로 둠, 참고용)")
        for _, r in has.iterrows():
            w(f"      {r['node_id']} [{classify_location(r['loc_2025'])}] {r['loc_2025']}")
    w("  - 위치정보는 텍스트 설명뿐이며 좌표가 없음 → 지오코딩 전에는 공간 위치가 확정되지 않음")

    w("\n10. 다음 단계")
    w("  - MATCHED 노드의 위치정보로 Kakao Local 주소/키워드 검색 → 좌표 후보 확보(이번 단계에서는 호출 안 함)")
    w("  - 지오코딩 결과가 강남구 경계 안에 있는지 검증, 실패 건은 수동 검토 목록으로 분리")
    w("  - 매칭률이 부족하면 python -m src.data.analyze_location_match 서초 동작 으로 동일 검증")
    return "\n".join(L) + "\n"


def main(districts: list[str]) -> None:
    period_stats, node_parts, per_month = [], [], {}
    for p in PERIOD_CSVS:
        stats, nodes = scan_period_file(p)
        period_stats.append(stats)
        node_parts.append(nodes)
        per_month[p.stem[-6:]] = nodes
    combos = pd.concat(node_parts)
    all_nodes = period_id_table(combos.groupby([c for c in combos.columns if c != "n_records"],
                                               dropna=False)["n_records"].sum().reset_index())
    loc2025 = None
    if LOC_COL in combos:
        l25 = combos.assign(node_id=combos[ID_COL].map(normalize_text),
                            loc_2025=combos[LOC_COL].map(normalize_text))
        loc2025 = l25.groupby("node_id")["loc_2025"].first().reset_index()

    master_stats, master = load_location_master()
    joined = join_status(all_nodes[["node_id", "district_code", "district_name"]], master)

    names = set(all_nodes["district_name"]) | set(master["district_name"])
    for d in districts:
        if d not in names:
            raise SystemExit(f"구분명 '{d}' 없음. 실제 값: {sorted(names)}")
        result = joined[joined["district_name"] == d].reset_index(drop=True)
        months = {m: set(n.loc[n[NAME_COL] == d, ID_COL].map(normalize_text)) for m, n in per_month.items()}
        slug = SLUGS.get(d) or result["district_code"].iloc[0]
        PROCESSED.mkdir(parents=True, exist_ok=True)
        result.to_csv(PROCESSED / f"{slug}_location_match.csv", index=False, encoding="utf-8-sig")
        report = build_report(d, period_stats, all_nodes, months, master_stats, master, joined, result, loc2025)
        (PROCESSED / f"{slug}_location_match_report.txt").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main(sys.argv[1:] or ["강남"])
