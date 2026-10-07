"""실행: python -m tests.test_analyze_location_match  (pytest로도 실행 가능)"""
import pandas as pd

from src.data.analyze_location_match import classify_location, join_status


def test_classify_location():
    assert classify_location("강남구 강남대로 435 맨홀(역삼역)") == "주소 형태"
    assert classify_location("서울특별시 강남구 역삼동 820-5") == "주소 형태"
    assert classify_location("서울 강남구 테헤란로 138") == "주소 형태"
    assert classify_location("양재동 82-8 앞 바우뫼로") == "도로명/지번 형태"
    assert classify_location("대치사거리 맨홀") == "랜드마크/시설물 설명"
    assert classify_location("???") == "기타/불명확"
    assert classify_location("  ") == "빈 값"
    assert classify_location(None) == "빈 값"


def test_join_status():
    period = pd.DataFrame({"node_id": ["23-0001", "23-0002", "23-0003"],
                           "district_code": ["23"] * 3, "district_name": ["강남"] * 3})
    master = pd.DataFrame({"node_id": ["23-0001", "23-0002", "23-0099"],
                           "district_code": ["23"] * 3, "district_name": ["강남"] * 3,
                           "location_info": ["강남구 강남대로 1", None, "강남구 테헤란로 2"]})
    r = join_status(period, master).set_index("node_id")
    assert r.loc["23-0001", "match_status"] == "MATCHED"
    assert r.loc["23-0002", "match_status"] == "NO_LOCATION"  # 존재하지만 위치정보 빈 값
    assert r.loc["23-0003", "match_status"] == "NO_LOCATION"  # 현재 파일에 없음
    assert r.loc["23-0099", "match_status"] == "LOCATION_ONLY"
    assert not r.loc["23-0099", "exists_in_2025_06_08"]
    assert r.loc["23-0099", "district_name"] == "강남"


if __name__ == "__main__":
    test_classify_location()
    test_join_status()
    print("ok")
