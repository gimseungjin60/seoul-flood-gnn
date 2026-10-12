"""실행: python -m tests.test_geocode_retry  (pytest로도 실행 가능). 실제 API는 호출하지 않는다."""
import pandas as pd

import src.data.geocode_retry as gr

SHOP = {"place_name": "우락뽈락", "address_name": "서울 강남구 삼성동 1", "road_address_name": "서울 강남구 삼성로148길 5",
        "x": "127.05", "y": "37.51"}


def test_candidates_rules():
    other_gu = {**SHOP, "address_name": "서울 서초구 서초동 1"}
    other_road = {**SHOP, "road_address_name": "서울 강남구 테헤란로 1"}
    other_name = {**SHOP, "place_name": "다른가게"}
    docs = [SHOP, other_gu, other_road, other_name]
    assert gr.candidates(docs, "강남구", "우락뽈락", "삼성로148길") == [SHOP]


def test_retry_node_outcomes():
    gr.REQUEST_INTERVAL_S = 0
    plan = (["q1", "q2"], "우락뽈락", "삼성로148길")
    ok = gr.retry_node("강남구", plan, "k", search=lambda q, m, k: [] if q == "q1" else [SHOP])
    assert ok["status"] == "OK" and ok["query"] == "q2" and ok["lat"] == 37.51
    multi = gr.retry_node("강남구", plan, "k", search=lambda q, m, k: [SHOP, SHOP])
    assert multi["status"] == "MULTIPLE" and "lat" not in multi  # 첫 번째 자동 선택 안 함
    none = gr.retry_node("강남구", plan, "k", search=lambda q, m, k: [{**SHOP, "place_name": "x"}])
    assert none["status"] == "NOT_FOUND"


def test_apply_retries_only_touches_targets():
    res = pd.DataFrame({"node_id": ["A", "B", "C"], "search_keyword": ["a", "b", "c"],
                        "geocoding_status": ["OK", "NOT_FOUND", "NOT_FOUND"],
                        "geocoding_method": ["ADDRESS", "KEYWORD", "KEYWORD"],
                        "latitude": [37.0, None, None], "longitude": [127.0, None, None],
                        "n_results": [1, 0, 0], "matched_address": ["x", None, None]})
    out = gr.apply_retries(res, {
        "B": {"status": "OK", "lat": 37.5, "lon": 127.5, "query": "q", "addr": "addr", "place": "p", "n": 1},
        "C": {"status": "MULTIPLE", "query": "q", "n": 3}}).set_index("node_id")
    assert (out.loc["A", "latitude"], out.loc["A", "location_basis"], out.loc["A", "needs_review"]) == (37.0, "ADDRESS", False)
    assert out.loc["B", "location_basis"] == "FACILITY_PROXY" and out.loc["B", "needs_review"]
    assert out.loc["B", "geocoding_method"] == gr.RETRY_METHOD
    assert out.loc["C", "geocoding_status"] == "MULTIPLE" and pd.isna(out.loc["C", "latitude"])


if __name__ == "__main__":
    test_candidates_rules()
    test_retry_node_outcomes()
    test_apply_retries_only_touches_targets()
    print("ok")
