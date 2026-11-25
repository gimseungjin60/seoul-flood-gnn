"""실행: python -m tests.test_t1_oracle  (pytest로도 실행 가능)
미래 강우 상한 테스트: 미래 강우 열은 블록 t+1..t+h 만 담고(목표 시각 이후 강우 없음), 격자 끝에서도 오류 없이 0."""
import numpy as np

from src.analysis.t1_oracle import future_rain_features
from src.analysis.t1_oracle_rules import o1_verdict, o2_verdict


def test_future_rain_window():
    rain = np.arange(20, dtype=float)
    rain[7] = np.nan
    pos = np.array([3, 5, 18])
    f = future_rain_features(rain, pos, 3)
    assert f.shape == (3, 3)
    assert np.allclose(np.expm1(f[0]), [4, 5, 6])              # t=3 → 블록 4,5,6 (t 자신과 7 이후는 없음)
    assert np.allclose(np.expm1(f[1]), [6, 0, 8])              # 결측(7) → 0
    assert np.allclose(np.expm1(f[2]), [19, 0, 0])             # 격자 밖 → 0
    assert np.allclose(np.expm1(future_rain_features(rain, pos, 1)[:, 0]), [4, 6, 19])


def test_verdict_labels():
    assert o1_verdict(-2, -1) == "미래 강우를 알면 오차가 줄어듦" and o1_verdict(-1, 1) == "구분 불가"
    assert o2_verdict(0.1, 0.3) == "미래 강우를 알면 큰 상승을 더 잡음" and o2_verdict(-0.1, 0.2) == "구분 불가"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
