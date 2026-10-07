"""실행: python -m tests.test_analyze_gangnam_quality  (pytest로도 실행 가능)"""
import pandas as pd

from src.data.analyze_gangnam_quality import grade, node_quality


def test_node_quality():
    start, end = pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-01 01:40")  # 100분
    t_a = pd.date_range(start, periods=100, freq="min")          # 60s 간격, 100% 커버
    t_b = pd.date_range(start, periods=10, freq="5min")          # 300s 간격, 10/20 = 50%
    df = pd.DataFrame({
        "고유번호": ["A"] * 100 + ["B"] * 10 + ["B"],
        "구분코드": "23", "구분명": "강남", "위치정보": "강남구 테헤란로 1",
        "측정일자": "x", "측정수위": ["0.1"] * 110 + [None], "통신상태": "통신양호",
        "time": list(t_a) + list(t_b) + [t_b[0]],                 # B: 중복 시각 1건 + 수위 결측 1건
    })
    q = node_quality(df, start, end).set_index("node_id")
    assert q.loc["A", "measurement_interval"] == "60s"
    assert q.loc["A", "coverage_ratio"] == 1.0 and q.loc["A", "quality_grade"] == "HIGH"
    assert q.loc["B", "measurement_interval"] == "300s"
    assert q.loc["B", "expected_count"] == 20 and q.loc["B", "measurement_count"] == 10
    assert q.loc["B", "quality_grade"] == "MEDIUM"
    assert q.loc["B", "n_dup_time"] == 1 and q.loc["B", "n_missing_or_blank"] == 1


def test_grade():
    assert [grade(x) for x in (0.8, 0.79, 0.5, 0.49)] == ["HIGH", "MEDIUM", "MEDIUM", "LOW"]


if __name__ == "__main__":
    test_node_quality()
    test_grade()
    print("ok")
