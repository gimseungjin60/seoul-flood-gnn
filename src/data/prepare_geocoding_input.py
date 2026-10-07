"""강남 6~8월 HIGH 노드 → Kakao 지오코딩 입력 파일 준비 (API 호출 없음, 원본은 읽기만 한다).

실행: python -m src.data.prepare_geocoding_input
입력: data/processed/district_quality_nodes.csv (6~8월 커버리지, analyze_district_quality 결과)
      data/processed/gangnam_202508_quality.csv (8월 커버리지, 교차 확인용)
      data/raw/tv_swm_wal_mea_2025{06,07,08}.csv (위치정보 원문)
출력: data/processed/gangnam_geocoding_input.csv, gangnam_geocoding_input_report.txt

- search_keyword = 2025 원본 위치정보 문자열 그대로(공백 정리·HTML 엔티티 복원 등 일절 안 함).
- 3개월 원문이 노드마다 1종이 아니면 임의로 고르지 않고 중단한다.
"""
import pandas as pd

from src.data.analyze_location_match import EN_TO_KO, LOC_COL, PERIOD_CSVS
from src.data.clean_nodes import ID_COL, PROCESSED
from src.data.inspect_water_level import read_chunks

DISTRICT_NAME = "강남"
NODES_CSV = PROCESSED / "district_quality_nodes.csv"
AUG_CSV = PROCESSED / "gangnam_202508_quality.csv"
OUT_CSV = PROCESSED / "gangnam_geocoding_input.csv"
OUT_REPORT = PROCESSED / "gangnam_geocoding_input_report.txt"
COLUMNS = ["node_id", "district_code", "district_name", "search_keyword", "location_source",
           "geocoding_status", "geocoding_method", "latitude", "longitude"]
_WANTED = {ID_COL, LOC_COL, "unq_no", "pstn_info"}


def high_nodes(nodes: pd.DataFrame, district: str = DISTRICT_NAME) -> pd.DataFrame:
    return nodes[(nodes["district_name"] == district) & (nodes["coverage_ratio"] >= 0.8)].sort_values("node_id")


def raw_locations(node_ids: set[str], paths=PERIOD_CSVS) -> pd.DataFrame:
    """원본 3개 파일에서 해당 노드의 위치정보 원문을 (node_id, month, 위치정보, 행 수)로 모은다."""
    parts = []
    for p in paths:
        for chunk in read_chunks(p, usecols=lambda c: c in _WANTED):
            chunk = chunk.rename(columns=EN_TO_KO)
            c = chunk[chunk[ID_COL].isin(node_ids)]
            parts.append(c.groupby([ID_COL, LOC_COL], dropna=False).size().rename("n").reset_index()
                         .assign(month=p.stem[-6:]))
    return (pd.concat(parts).groupby([ID_COL, "month", LOC_COL], dropna=False)["n"].sum().reset_index())


def build_input(high: pd.DataFrame, locs: pd.DataFrame) -> pd.DataFrame:
    variants = locs.groupby(ID_COL)[LOC_COL].agg(lambda s: list(s.dropna().unique()))
    bad = variants[variants.map(len) != 1]
    if len(bad):
        raise SystemExit(f"위치정보 원문이 1종이 아닌 노드(임의 선택 안 함): {bad.to_dict()}")
    out = pd.DataFrame({
        "node_id": high["node_id"].to_numpy(),
        "district_code": high["district_code"].astype(str).str.zfill(2).to_numpy(),
        "district_name": high["district_name"].to_numpy(),
    })
    out["search_keyword"] = out["node_id"].map(variants.str[0])
    out["location_source"] = "2025_original_location"
    out["geocoding_status"] = "PENDING"
    out["geocoding_method"] = "NONE"
    out["latitude"] = None
    out["longitude"] = None
    return out[COLUMNS]


def build_report(out: pd.DataFrame, high: pd.DataFrame, aug_high: set[str], locs: pd.DataFrame) -> str:
    L = []
    w = L.append
    w(f"[{DISTRICT_NAME}] Kakao 지오코딩 입력 준비 리포트 (API 호출 없음)")
    w("=" * 60)
    w(f"대상: 2025-06~08 커버리지 >= 80% (HIGH) 노드 {len(out)}개  ← {NODES_CSV.name}")
    same = set(out["node_id"]) == aug_high
    w(f"8월 품질 파일({AUG_CSV.name})의 HIGH와 동일 집합 여부: {same}"
      + ("" if same else f" (6~8월만 {sorted(set(out['node_id']) - aug_high)}, "
                         f"8월만 {sorted(aug_high - set(out['node_id']))})"))
    w("위치정보 출처: " + ", ".join(f"data/raw/{p.name}" for p in PERIOD_CSVS) + " 의 위치정보 컬럼 원문")
    months = locs.groupby(ID_COL)["month"].nunique()
    w(f"3개월 모두 같은 위치정보 원문 1종: {int((months == 3).sum())}/{len(out)} 노드")
    w("\n노드별 search_keyword (원문 그대로)")
    cov = high.set_index("node_id")["coverage_ratio"]
    for _, r in out.iterrows():
        w(f"  {r['node_id']}  cov {cov[r['node_id']]:.1%}  {r['search_keyword']}")
    w("\n지오코딩 시 주의할 원문 특성 (수정하지 않음, 표시만)")
    kw = out.set_index("node_id")["search_keyword"]
    flags = {
        "HTML 엔티티(&lt; 등) 포함": kw[kw.str.contains(r"&lt;|&gt;|&amp;")],
        "괄호/부가설명 포함": kw[kw.str.contains(r"[(<]")],
        "'앞/옆/맨홀/도로' 등 위치 설명어 포함": kw[kw.str.contains(r"앞|옆|맨홀|도로|보도")],
        "구 이름 없이 시작": kw[~kw.str.contains("강남구")],
    }
    for k, v in flags.items():
        w(f"  - {k}: {len(v)} {list(v.index)}")
    w("\n다음 단계")
    w("  1. .env에 KAKAO_REST_API_KEY 설정")
    w("  2. python -m src.data.geocoder  → gangnam_geocoding_result.csv (입력 파일은 그대로 둠)")
    w("  3. 결과가 정확히 1건이고 강남구 안일 때만 좌표 기록, 나머지는 상태만 남기고 수동 검토")
    return "\n".join(L) + "\n"


def main() -> None:
    nodes = pd.read_csv(NODES_CSV, dtype={"district_code": str})
    high = high_nodes(nodes)
    aug = pd.read_csv(AUG_CSV)
    aug_high = set(aug.loc[aug["quality_grade"] == "HIGH", "node_id"])
    locs = raw_locations(set(high["node_id"]))
    out = build_input(high, locs)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    report = build_report(out, high, aug_high, locs)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
