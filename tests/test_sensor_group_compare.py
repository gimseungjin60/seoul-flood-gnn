"""sensor_group_compare 셀프 테스트 (원본 CSV 없이 합성 데이터)."""
import pandas as pd

from src.analysis.sensor_group_compare import address_class, dry_mask, is_jibun_only, node_stats


def test_address_class():
    assert address_class("강남구 강남대로122길 17") == "대로"
    assert address_class("강남구 논현로 71길 16") == "로+길/길"
    assert address_class("강남구 삼성로212앞 맨홀") == "로"
    assert address_class("강남구 일원동 681-4") == "도로명 없음"
    assert is_jibun_only("강남구 일원동 681-4")
    assert not is_jibun_only("강남구 논현로 71길 16")


def test_node_stats_interval_and_coverage():
    t = pd.date_range("2025-08-05", periods=5, freq="1min")
    d = pd.DataFrame({"t": t, "v": [0.1, 0.2, 0.3, 0.2, 0.1]})
    s = node_stats(d)
    assert s["n"] == 5 and s["interval_med_s"] == 60 and s["max"] == 0.3
    assert 0 < s["coverage"] < 0.01


def test_dry_mask_excludes_event_with_margin():
    t = pd.Series(pd.to_datetime(["2025-08-13 02:00", "2025-08-13 03:30", "2025-08-14 12:00", "2025-08-14 15:00"]))
    ev = pd.DataFrame({"start": [pd.Timestamp("2025-08-13 04:30")], "end": [pd.Timestamp("2025-08-14 11:29")]})
    m = dry_mask(t, ev)
    assert list(m) == [True, False, False, True]  # 시작 1시간 전부터, 끝 3시간 뒤까지 제외


if __name__ == "__main__":
    test_address_class()
    test_node_stats_interval_and_coverage()
    test_dry_mask_excludes_event_with_margin()
    print("test_sensor_group_compare: OK")
