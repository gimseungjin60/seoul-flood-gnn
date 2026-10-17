"""실행: python -m tests.test_prepare_rainfall  (pytest로도 실행 가능). 실제 데이터 파일 불필요(임시 CSV 사용)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare_rainfall import find_events, lag_corr, read_raw, to_grid


def _write(rows):
    p = Path(tempfile.mkdtemp()) / "r.csv"
    df = pd.DataFrame(rows, columns=["강우량계 코드", "강우량계명", "구청 코드", "구청명", "10분우량", "자료수집 시각"])
    df.to_csv(p, index=False, encoding="cp949")
    return [p]


def test_slot_label_is_interval_end_and_jitter_rows_map_to_same_slot():
    rows = [("1", "A", "1", "강남구", "0.5", "2025-06-01 00:09"),
            ("1", "A", "1", "강남구", "1", "2025-06-01 00:19:59"),      # 초 포함 → 같은 :19 구간
            ("1", "A", "1", "강남구", "2", "2025-06-01 00:28"),         # 어긋난 시각 → :29 구간
            ("1", "A", "1", "강남구", "0", "2025-06-01 00:39")]
    d = read_raw(_write(rows))
    assert d["t"].dt.strftime("%H:%M").tolist() == ["00:09", "00:19", "00:29", "00:39"]
    times, gids, info, values = to_grid(d)
    assert len(times) == 4 and gids == ["1"] and values[:, 0].tolist() == [0.5, 1.0, 2.0, 0.0]


def test_missing_slot_is_nan_not_zero_and_duplicates_are_rejected():
    rows = [("1", "A", "1", "강남구", "1", "2025-06-01 00:09"), ("1", "A", "1", "강남구", "1", "2025-06-01 00:29")]
    times, _, _, values = to_grid(read_raw(_write(rows)))
    assert np.isnan(values[1, 0]) and values[0, 0] == 1.0               # 00:19 행이 없으면 NaN (0mm 아님)
    dup = rows + [("1", "A", "1", "강남구", "3", "2025-06-01 00:09")]
    try:
        to_grid(read_raw(_write(dup)))
    except ValueError:
        return
    raise AssertionError("같은 지점·구간 중복이 거부되지 않음")


def test_events_split_by_dry_gap_and_nan_counts_as_dry():
    times = pd.date_range("2025-06-01 00:09", periods=200, freq="10min")
    rain = np.zeros(200)
    rain[[2, 3]] = [1.0, 2.0]
    rain[60] = 5.0                         # 56칸(>36) 뒤 → 별도 이벤트
    rain[80] = 1.0                         # 20칸 뒤 → 같은 이벤트
    rain[100:110] = np.nan                 # 결측은 무강우로 취급(채우지 않음)
    ev = find_events(rain, times)
    assert len(ev) == 2 and ev.total_mm.tolist() == [3.0, 6.0] and ev.max_10min_mm.tolist() == [2.0, 5.0]
    assert ev.start[0] == times[2] - pd.Timedelta(minutes=9)           # 시작은 구간의 시작 시각


def test_lag_corr_finds_known_response_delay():
    idx = pd.date_range("2025-06-01 00:09", periods=3000, freq="10min")
    rng = np.random.default_rng(0)
    rain = pd.Series(np.where(rng.random(3000) < 0.1, rng.uniform(1, 5, 3000), 0.0), index=idx)
    net = pd.Series(np.cumsum(rain.shift(2, fill_value=0.0).to_numpy()), index=idx)   # 수위 변화 = 강우의 20분 뒤
    lc = lag_corr(rain, net)
    assert int(lc.loc[lc["corr"].idxmax(), "lag_min"]) == 20


if __name__ == "__main__":
    test_slot_label_is_interval_end_and_jitter_rows_map_to_same_slot()
    test_missing_slot_is_nan_not_zero_and_duplicates_are_rejected()
    test_events_split_by_dry_gap_and_nan_counts_as_dry()
    test_lag_corr_finds_known_response_delay()
    print("ok")
