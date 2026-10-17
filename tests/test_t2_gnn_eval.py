"""실행: python -m tests.test_t2_gnn_eval. 사전 고정 판정 규칙(3단계, 보조 조건)이 문서대로 동작하는지만 본다."""
from src.analysis.t2_gnn_eval import primary_verdict, secondary_ok


def test_primary_three_way_rule():
    assert primary_verdict(-2.0, -0.1) == "GNN이 IDW를 넘음"
    assert primary_verdict(-1.0, 0.0) == "구분 불가"          # 상한이 정확히 0 이면 '0보다 작지' 않다
    assert primary_verdict(-1.0, 1.0) == "구분 불가"
    assert primary_verdict(0.1, 2.0) == "IDW보다 나쁨"
    assert primary_verdict(0.0, 2.0) == "구분 불가"


def test_secondary_requires_primary_and_upper_below_zero():
    assert secondary_ok("GNN이 IDW를 넘음", -0.1)
    assert not secondary_ok("GNN이 IDW를 넘음", 0.0)
    assert not secondary_ok("구분 불가", -1.0)
    assert not secondary_ok("IDW보다 나쁨", -1.0)


if __name__ == "__main__":
    test_primary_three_way_rule()
    test_secondary_requires_primary_and_upper_below_zero()
    print("test_t2_gnn_eval: OK")
