"""실행: python -m tests.test_clean_nodes  (pytest로도 실행 가능)"""
import pandas as pd

from src.data.clean_nodes import build_nodes, normalize_text


def test_normalize_text():
    assert normalize_text("  01-0001\r\n") == "01-0001"
    assert normalize_text("동대문　 \xa0구") == "동대문 구"
    assert normalize_text("​") is None
    assert normalize_text("OO아파트 옆") == "OO아파트 옆"  # 의미 단어는 유지


def test_build_nodes_flags():
    combos = pd.DataFrame({
        "raw_node_id":       ["01-0001", "02-0001", "02-0001", " 03-0001", "05-0001"],
        "raw_district_code": ["01",      "02",      "03",      "03",       "04"],
        "raw_district_name": ["종로",    "중",      "용산",    "용산",     "성동"],
        "n_records": [10, 5, 5, 7, 3],
        "first_measured": ["2026-09-01"] * 5,
        "last_measured": ["2026-09-30"] * 5,
    })
    n = build_nodes(combos).set_index("node_id")
    assert len(n) == 4
    assert not n.loc["01-0001", "needs_review"]
    assert "혼재" in n.loc["02-0001", "review_reason"]
    assert n.loc["02-0001", "n_records"] == 10
    assert "공백" in n.loc["03-0001", "review_reason"]
    assert "접두부" in n.loc["05-0001", "review_reason"]


if __name__ == "__main__":
    test_normalize_text()
    test_build_nodes_flags()
    print("ok")
