"""1차 지오코딩 NOT_FOUND 노드를 원문 속 시설명으로 Kakao 키워드 재검색 (결과 파일의 해당 행만 갱신).

실행: python -m src.data.geocode_retry
갱신: data/processed/gangnam_geocoding_result.csv (재검색 대상 행만, 나머지 행 좌표는 변경 금지 검증)
출력: data/processed/gangnam_geocoding_retry_report.txt

정확성 우선 규칙 — 후보 = 키워드 결과 중 (1) 주소가 대상 구 (2) 장소명에 원문 시설명 포함
(3) 원문에 도로명이 있으면 도로명주소에 그 도로명 포함. 후보가 정확히 1개일 때만 좌표 기록.
후보가 2개 이상이면 MULTIPLE(좌표 없음), 모든 검색어에서 0개면 NOT_FOUND 유지.
좌표는 '시설' 위치이지 맨홀 위치가 아니므로 location_basis=FACILITY_PROXY, needs_review=True로 표시한다.
"""
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

from src.data.geocoder import OUT_CSV as RESULT_CSV
from src.data.geocoder import REQUEST_INTERVAL_S, kakao_search
from src.data.clean_nodes import PROCESSED
from src.data.inspect_water_level import PROJECT_ROOT

REPORT = PROCESSED / "gangnam_geocoding_retry_report.txt"
RETRY_METHOD = "KEYWORD_FACILITY_RETRY"
# node_id → (검색어 목록(순서대로), 장소명에 있어야 할 원문 시설명, 도로명주소에 있어야 할 원문 도로명 or None)
RETRY_PLAN = {
    "23-0007": (["강남구 윤호병원", "윤호병원"], "윤호병원", None),
    "23-0011": (["삼성로148길 우락뽈락", "강남구 우락뽈락"], "우락뽈락", "삼성로148길"),
    # 원문은 '개포로 140길'(도로) — '개포로140'은 건물번호로 오인될 수 있어 시설명 일치를 반드시 요구
    "23-0014": (["개포로140", "개포로140길 현대블루핸즈", "강남구 현대블루핸즈 개포"], "블루핸즈", "개포로"),
    "23-0018": (["일원본동 주민센터", "일원본동주민센터"], "일원본동", None),
}


def _nospace(s) -> str:
    return "".join(str(s or "").split())


def candidates(docs: list[dict], district_full: str, name: str, road: str | None) -> list[dict]:
    return [d for d in docs
            if district_full in (d.get("address_name") or "").split()
            and name in _nospace(d.get("place_name"))
            and (road is None or _nospace(road) in _nospace(d.get("road_address_name")))]


def retry_node(district_full: str, plan, api_key: str, search=kakao_search) -> dict:
    """검색어를 순서대로 시도. 후보 1개 → OK, 2개 이상 → MULTIPLE, 전부 0개 → NOT_FOUND. 시도 기록 포함."""
    queries, name, road = plan
    log = []
    for q in queries:
        docs = search(q, "KEYWORD", api_key)
        time.sleep(REQUEST_INTERVAL_S)
        cands = candidates(docs, district_full, name, road)
        log.append({"query": q, "n_docs": len(docs), "n_candidates": len(cands),
                    "docs": [(d.get("place_name"), d.get("road_address_name") or d.get("address_name"))
                             for d in docs[:5]]})
        if len(cands) == 1:
            d = cands[0]
            return {"status": "OK", "lat": float(d["y"]), "lon": float(d["x"]), "query": q,
                    "addr": d.get("road_address_name") or d.get("address_name"),
                    "place": d.get("place_name"), "n": len(docs), "log": log}
        if len(cands) > 1:
            return {"status": "MULTIPLE", "query": q, "n": len(docs), "log": log}
    return {"status": "NOT_FOUND", "query": queries[-1], "n": 0, "log": log}


def apply_retries(res: pd.DataFrame, outcomes: dict[str, dict]) -> pd.DataFrame:
    out = res.copy()
    for col, default in (("location_basis", None), ("needs_review", False),
                         ("retry_query", None), ("matched_place", None)):
        if col not in out:
            out[col] = default
    ok_addr = out["geocoding_status"] == "OK"
    out.loc[ok_addr & out["location_basis"].isna(), "location_basis"] = "ADDRESS"
    for node, o in outcomes.items():
        i = out.index[out["node_id"] == node][0]
        out.loc[i, ["geocoding_status", "geocoding_method", "retry_query", "n_results"]] = \
            [o["status"], RETRY_METHOD, o["query"], o["n"]]
        out.loc[i, "needs_review"] = True
        if o["status"] == "OK":
            out.loc[i, ["latitude", "longitude", "matched_address", "matched_place", "location_basis"]] = \
                [o["lat"], o["lon"], o["addr"], o["place"], "FACILITY_PROXY"]
    return out


def build_report(before: pd.DataFrame, after: pd.DataFrame, outcomes: dict[str, dict]) -> str:
    L = []
    w = L.append
    w("[강남] 지오코딩 재검색 리포트 (NOT_FOUND 4개, 원문 시설명 키워드 검색)")
    w("=" * 60)
    w("채택 규칙: 결과 주소가 강남구 + 장소명에 원문 시설명 포함 + (원문 도로명 있으면) 도로명주소 일치,")
    w("           이 조건을 만족하는 후보가 정확히 1개일 때만 좌표 기록. 첫 번째 결과 자동 선택 없음.")
    for node, o in outcomes.items():
        kw = after.loc[after["node_id"] == node, "search_keyword"].iloc[0]
        w(f"\n{node}  원문: {kw}")
        for a in o["log"]:
            w(f"  검색어 '{a['query']}': 결과 {a['n_docs']}건, 규칙 통과 {a['n_candidates']}건")
            for p, ad in a["docs"]:
                w(f"      - {p} | {ad}")
        if o["status"] == "OK":
            w(f"  → OK (시설 위치 대용): {o['place']} | {o['addr']} | {o['lat']:.6f}, {o['lon']:.6f}")
        else:
            w(f"  → {o['status']} (좌표 없음, 검토 대기)")
    c = after["geocoding_status"].value_counts()
    w(f"\n최종 상태: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    w(f"  좌표 확보 {int(after['latitude'].notna().sum())}/{len(after)} "
      f"(주소 기반 {int((after['location_basis'] == 'ADDRESS').sum())}, "
      f"시설 대용 {int((after['location_basis'] == 'FACILITY_PROXY').sum())})")
    w(f"  검토 필요(needs_review): {', '.join(after.loc[after['needs_review'].astype(bool), 'node_id'])}")
    w("\n주의: FACILITY_PROXY 좌표는 원문에 적힌 시설(병원/가게/주민센터 등)의 위치이며, 수위계 맨홀은")
    w("      그 시설 '앞 보도/도로'에 있다고만 기술되어 있다. 맨홀 실제 위치와 수십 m 차이가 날 수 있다.")
    w(f"\n기존 OK {int((before['geocoding_status'] == 'OK').sum())}개 좌표 변경 여부: 변경 없음(저장 전 검증)")
    return "\n".join(L) + "\n"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        sys.exit(".env에 KAKAO_REST_API_KEY가 없습니다.")
    before = pd.read_csv(RESULT_CSV, dtype={"district_code": str})
    targets = before[(before["geocoding_status"] == "NOT_FOUND") & before["node_id"].isin(RETRY_PLAN)]
    outcomes = {}
    for _, r in targets.iterrows():
        try:
            outcomes[r["node_id"]] = retry_node(f"{r['district_name']}구", RETRY_PLAN[r["node_id"]], api_key)
        except Exception as e:  # 키 노출 방지: 예외 타입만 기록
            outcomes[r["node_id"]] = {"status": "ERROR", "query": None, "n": 0,
                                      "log": [{"query": "-", "n_docs": 0, "n_candidates": 0,
                                               "docs": [(type(e).__name__, "")]}]}
    after = apply_retries(before, outcomes)
    # 기존 성공 16개 좌표가 그대로인지 확인 후에만 저장
    ok = before["geocoding_status"] == "OK"
    pd.testing.assert_frame_equal(before.loc[ok, ["node_id", "latitude", "longitude"]],
                                  after.loc[ok, ["node_id", "latitude", "longitude"]])
    report = build_report(before, after, outcomes)
    tmp = RESULT_CSV.with_suffix(".tmp")
    after.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(RESULT_CSV)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
