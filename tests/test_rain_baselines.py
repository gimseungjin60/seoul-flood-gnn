"""실행: python -m tests.test_rain_baselines  (pytest로도 실행 가능). 실제 데이터 파일 불필요."""
import numpy as np
import pandas as pd

from src.models.rain_baselines import event_mask_for_targets, rain_features, standardize


def _grid(n=400):
    label = 9 + 10 * np.arange(n)                      # 슬롯 끝 분: 9, 19, 29, ... (수위 분 인덱스)
    rain = np.zeros(n)
    return label, rain


def test_only_completed_slots_are_used():
    label, rain = _grid()
    rain[200] = 5.0                                    # 슬롯 200: 분 2000~2009 (끝 라벨 2009)
    t_before = np.array([2009])                        # t=2009분: 슬롯 200은 2009:59에 끝남 → 아직 사용 불가
    t_after = np.array([2010])                         # t=2010분: 사용 가능
    f0 = rain_features(rain, label, t_before)
    f1 = rain_features(rain, label, t_after)
    assert f0[0, :6].sum() == 0.0                      # 마지막 완료 슬롯은 199, 200은 미완료 → 강우 특징 0
    assert np.isclose(f1[0, 0], np.log1p(5.0))        # 가장 최근 슬롯이 200


def test_missing_slots_are_zero_with_flag_not_fabricated():
    label, rain = _grid()
    rain[198] = np.nan
    f = rain_features(rain, label, np.array([2010]))   # 최근 6개 슬롯 = 200,199,198,197,196,195
    assert f[0, 6] == 1.0 and f[0, 2] == 0.0           # 결측 1개, 값은 0으로 두되 결측 수로 표시


def test_cumulative_windows():
    label, rain = _grid()
    rain[200 - 17: 201] = 1.0                          # 슬롯 183~200(18개) 각 1mm
    f = rain_features(rain, label, np.array([2010]))
    assert np.isclose(f[0, 7], np.log1p(18.0)) and np.isclose(f[0, 8], np.log1p(18.0))   # 3h=18슬롯, 6h 안에 같은 18개


def test_too_early_anchor_is_rejected():
    label, rain = _grid()
    try:
        rain_features(rain, label, np.array([300]))    # 24시간(144슬롯) 이전 자료가 없다
    except ValueError:
        return
    raise AssertionError("이전 슬롯 부족이 거부되지 않음")


def test_standardize_uses_train_stats_and_constant_columns_safe():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    f = standardize(x)
    z = f(np.array([[2.0, 5.0]]))
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0           # 상수 열은 0으로 나눠지지 않는다


def test_event_mask_window_includes_three_hours_after():
    s0 = pd.Timestamp("2025-06-01 00:00")
    ev = pd.DataFrame({"start": [pd.Timestamp("2025-06-01 01:00")], "end": [pd.Timestamp("2025-06-01 02:00")]})
    m = event_mask_for_targets(np.array([30, 61, 119, 300, 301]), s0, ev)
    assert m.tolist() == [False, True, True, True, False]


if __name__ == "__main__":
    test_only_completed_slots_are_used()
    test_missing_slots_are_zero_with_flag_not_fabricated()
    test_cumulative_windows()
    test_too_early_anchor_is_rejected()
    test_standardize_uses_train_stats_and_constant_columns_safe()
    test_event_mask_window_includes_three_hours_after()
    print("ok")
