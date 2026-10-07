"""실행: python -m tests.test_prepare_geocoding_input  (pytest로도 실행 가능)"""
import pandas as pd

from src.data.prepare_geocoding_input import COLUMNS, build_input, high_nodes


def test_build_input_keeps_raw_text():
    nodes = pd.DataFrame({"node_id": ["23-0001", "23-0099"], "district_code": ["23", "23"],
                          "district_name": ["강남", "강남"], "coverage_ratio": [0.99, 0.1]})
    high = high_nodes(nodes)
    raw = "강남구 압구정로20길 6앞 맨홀&lt;한스빌딩&gt;"
    locs = pd.DataFrame({"고유번호": ["23-0001"] * 3, "month": ["202506", "202507", "202508"],
                         "위치정보": [raw] * 3, "n": [1, 1, 1]})
    out = build_input(high, locs)
    assert list(out.columns) == COLUMNS and len(out) == 1
    r = out.iloc[0]
    assert r["search_keyword"] == raw  # 원문 그대로
    assert (r["geocoding_status"], r["geocoding_method"], r["location_source"]) == \
        ("PENDING", "NONE", "2025_original_location")
    assert pd.isna(r["latitude"]) and pd.isna(r["longitude"])


def test_build_input_refuses_conflicting_locations():
    high = pd.DataFrame({"node_id": ["23-0001"], "district_code": ["23"], "district_name": ["강남"]})
    locs = pd.DataFrame({"고유번호": ["23-0001"] * 2, "month": ["202506", "202507"],
                         "위치정보": ["A", "B"], "n": [1, 1]})
    try:
        build_input(high, locs)
        assert False, "should abort"
    except SystemExit:
        pass


if __name__ == "__main__":
    test_build_input_keeps_raw_text()
    test_build_input_refuses_conflicting_locations()
    print("ok")
