"""먼저 오르는 센서가 지형상 더 높은 곳에 있는가(센서 간 시차의 후속 분석)의 사전 고정 규칙과 순수 함수. 실행 코드는 lead_elev.py.

==================================================================================================================================
사전 고정 규칙 (2026-11-14, 이 분석의 값을 계산하기 전에 정함. 이후 변경 금지. 바꾸려면 새 파일로 만들고 "사후"로 표시한다)
==================================================================================================================================
[목적] 시차 분석(docs/20·21)에서 쌍의 시차는 대부분 센서별 반응 속도로 설명됐다(사후 상관 0.82). 이것이 관로 흐름(상류가 먼저)인지
  위치의 성질인지 구분하려고, 반응이 빠른 센서가 지형상 높은 곳에 있는지 본다. 모델 비교가 아니라 자료 진단이다.
[고정 전에 알고 있던 것 — 숨기지 않고 적는다]
  - docs/14: 센서 고도 9.5~61.2 m(중앙 21.4 m), 가장 높은 센서는 23-0017(61.2 m).
  - docs/21: 먼저 오르는 센서 23-0088·0089·0020·0008, 늦게 오르는 센서 23-0003·0007·0096·0017·0015 (목록 수준).
  - 다른 센서의 고도와 반응 속도-고도 관계는 계산하지 않았다.
[입력] gangnam2026_lead_lag_events.csv(docs/20 규칙으로 만든 센서-사건 표, 유효 여부·반응 지연 D),
  gangnam2026_lead_lag_pairs.csv(일관된 시차 쌍), gangnam2026_station_elevation.csv의 elev_m(GLO-30 3x3 최솟값, docs/14).
[센서 반응 속도] 사건마다 유효 센서의 D 에서 그 사건 유효 센서 D 의 중앙값을 뺀 값(사건 보정 지연)의 센서별 중앙값.
  유효 사건 4개 이상인 센서만. 작을수록 먼저 오른다.
[판정 — 이 두 가지뿐]
  E1 (주, 센서 단위) 반응 속도와 고도의 Spearman 순위 상관 ρ. 센서 복원추출 2000회(시드 0) 95% 구간.
     상한 < 0 → "높은 센서가 먼저 오름" / 0 포함 → "고도와 반응 순서의 관계 구분 불가" / 하한 > 0 → "낮은 센서가 먼저 오름".
  E2 (보조, 쌍 단위) 일관된 시차 쌍 중 고도 차 ≥ 2 m 인 쌍에서 앞선 센서가 더 높은 비율 p.
     영가설: 센서들의 고도를 서로 무작위로 섞는다(2000회, 시드 0; 센서 단위 섞기로 쌍 간 의존을 유지). 쌍 목록과 앞선 센서는 그대로.
     p > 영분포 97.5백분위 → "앞선 센서가 더 높은 쪽" / p < 2.5백분위 → "앞선 센서가 더 낮은 쪽" / 그 사이 → "구분 불가".
[참고 (판정에 쓰지 않음)] 도로 이웃의 일관된 쌍만으로 E2 비율, 23-0017 제외 시 E1, 같은 위치인 23-0088·0089 중 하나만 남긴 E1.
[한계] 30 m 표면 고도는 관로 매설 깊이·경사가 아니다. 강남은 대부분 15~25 m 평지라 고도 차가 잡음 수준일 수 있다. 관로는 지형을
  따르지 않고 펌프장·간선으로 이어질 수 있다. 센서 수가 적다(유효 사건 4개 이상 센서만).
==================================================================================================================================
"""
import numpy as np
from scipy.stats import spearmanr

FREEZE_DATE = "2026-11-14"
MIN_EVENTS = 4
ELEV_DIFF_MIN = 2.0
N_BOOT = 2000
N_PERM = 2000
SEED = 0
E1_NEG, E1_ZERO, E1_POS = "높은 센서가 먼저 오름", "고도와 반응 순서의 관계 구분 불가", "낮은 센서가 먼저 오름"
E2_HI, E2_ZERO, E2_LO = "앞선 센서가 더 높은 쪽", "구분 불가", "앞선 센서가 더 낮은 쪽"


def sensor_speed(event_id: np.ndarray, node_id: np.ndarray, delay: np.ndarray, min_events: int = MIN_EVENTS) -> dict:
    """유효 센서-사건의 (사건, 센서, D) → {센서: 사건 보정 지연의 중앙값} (유효 사건 min_events 이상)."""
    event_id, node_id, delay = map(np.asarray, (event_id, node_id, delay))
    centered = delay.astype(float).copy()
    for e in np.unique(event_id):
        m = event_id == e
        centered[m] -= np.median(delay[m])
    out = {}
    for n in np.unique(node_id):
        m = node_id == n
        if m.sum() >= min_events:
            out[str(n)] = float(np.median(centered[m]))
    return out


def rho_ci(speed: np.ndarray, elev: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    rho = spearmanr(speed, elev).statistic
    rng = np.random.default_rng(seed)
    n = len(speed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.unique(speed[i]).size > 1 and np.unique(elev[i]).size > 1:
            bs.append(spearmanr(speed[i], elev[i]).statistic)
    return float(rho), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def e1_verdict(lo: float, hi: float) -> str:
    return E1_NEG if hi < 0 else (E1_POS if lo > 0 else E1_ZERO)


def leader_higher_frac(pairs: list[tuple[str, str]], elev: dict, min_diff: float = ELEV_DIFF_MIN) -> tuple[float, int]:
    """(앞선 센서, 뒤따르는 센서) 쌍 중 고도 차 ≥ min_diff 인 쌍에서 앞선 센서가 높은 비율과 쌍 수."""
    d = np.array([elev[a] - elev[b] for a, b in pairs])
    d = d[np.abs(d) >= min_diff]
    return (float(np.mean(d > 0)) if len(d) else float("nan")), int(len(d))


def elev_shuffle_null(pairs, elev: dict, n_perm: int = N_PERM, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys = sorted(elev)
    vals = np.array([elev[k] for k in keys])
    out = np.empty(n_perm)
    for p in range(n_perm):
        out[p] = leader_higher_frac(pairs, dict(zip(keys, rng.permutation(vals))))[0]
    return out


def e2_verdict(p: float, null: np.ndarray) -> str:
    lo, hi = np.nanpercentile(null, [2.5, 97.5])
    return E2_HI if p > hi else (E2_LO if p < lo else E2_ZERO)
