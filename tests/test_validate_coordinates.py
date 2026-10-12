"""실행: python -m tests.test_validate_coordinates  (pytest로도 실행 가능)"""
import tempfile
from pathlib import Path

import pandas as pd

from src.data.validate_coordinates import build_map, build_report, check_coordinates, marker_ids


def sample() -> pd.DataFrame:
    base = {"search_keyword": "x", "geocoding_status": "OK", "matched_place": None}
    rows = [
        ("A", 37.5000, 127.0500, "ADDRESS", False, "서울 강남구 a"),
        ("B", 37.5100, 127.0600, "FACILITY_PROXY", True, "서울 강남구 b"),
        ("C", 37.5000, 127.0505, "ADDRESS", False, "서울 강남구 c"),      # A와 약 44 m → 근접
        ("D", None, None, None, True, None),                               # 좌표 없음
        ("E", 35.1000, 129.0000, "ADDRESS", False, "부산 해운대구 e"),     # 강남 범위 밖
        ("F", "abc", 127.0, "ADDRESS", False, "서울 강남구 f"),            # 숫자 아님
        ("G", 95.0, 127.0, "ADDRESS", False, "서울 강남구 g"),             # 위도 범위 밖
    ]
    return pd.DataFrame([{**base, "node_id": n, "latitude": la, "longitude": lo, "location_basis": b,
                          "needs_review": rv, "matched_address": ad} for n, la, lo, b, rv, ad in rows])


def test_checks_detect_bad_coordinates():
    d, pairs = check_coordinates(sample())
    d = d.set_index("node_id")
    assert d.loc["F", "non_numeric"] and not d.loc["F", "valid"]
    assert d.loc["G", "out_of_range"] and not d.loc["G", "valid"]
    assert d.loc["D", "missing"] and not d.loc["D", "valid"]
    assert d.loc["E", "outside_bbox"] and d.loc["E", "address_not_gangnam"]
    assert d.loc["A", "near"] and d.loc["C", "near"] and not d.loc["A", "duplicate"]
    assert sorted(d.index[d["valid"]]) == ["A", "B", "C", "E"]


def test_map_shows_only_valid_and_distinguishes_basis():
    d, _ = check_coordinates(sample())
    m = build_map(d)
    assert sorted(marker_ids(m)) == ["A", "B", "C", "E"]  # D/F/G(좌표 없음·잘못됨)는 마커 없음
    cls = d.set_index("node_id")["marker_class"]
    assert cls["B"] == "FACILITY_PROXY"
    assert cls["A"] == "REVIEW" and cls["E"] == "REVIEW"  # 근접 / 범위 밖 경고
    clean = check_coordinates(sample().query("node_id in ['A', 'B']"))[0].set_index("node_id")["marker_class"]
    assert (clean["A"], clean["B"]) == ("ADDRESS", "FACILITY_PROXY")


def test_report_is_written():
    d, pairs = check_coordinates(sample())
    report = build_report(d, pairs, len(marker_ids(build_map(d))))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.txt"
        p.write_text(report, encoding="utf-8")
        text = p.read_text(encoding="utf-8")
    for s in ("전체 대상 지점: 7", "좌표 확보 지점: 4", "지도에 표시된 마커: 4 (유효 좌표 수와 일치: True)",
              "시설 위치 대용 좌표: 1", "근접 쌍: 1"):
        assert s in text, s


if __name__ == "__main__":
    test_checks_detect_bad_coordinates()
    test_map_shows_only_valid_and_distinguishes_basis()
    test_report_is_written()
    print("ok")
