"""센서 간 시차 분석(비가 올 때 한 센서가 먼저 오르고 다른 센서가 뒤따르는가)의 사전 고정 규칙과 순수 함수. 실행 코드는 lead_lag.py.

==================================================================================================================================
사전 고정 규칙 (2026-11-14, 이 분석의 어떤 값도 계산하기 전에 정함. T1 시공간 GNN 판정(t1_st_eval)도 아직 실행 전. 이후 변경 금지.
바꾸려면 새 파일로 별도 분석을 만들고 "사후"로 표시한다)
==================================================================================================================================
[목적] T1(docs/17)은 "위쪽 물이 몇십 분 뒤 아래로 내려온다"는 시간차를 그래프가 전달해 준다고 가정한다. 그 시간차가 자료에
  실제로 있는지 모델 없이 확인한다. 모델 비교가 아니라 자료 진단이며, T1 판정에는 쓰지 않는다.
[데이터] gangnam2026_station_timeseries.npz 1분 수위(품질 규칙 R1·R2, 보간 없음), 센서 = 2026 pool 48개(load_roles()["pool"], J1·J2 적용),
  평가 이벤트 13개(is_eval). 강우 = 각 센서의 최근접 강우량계(gangnam2026 배정) 10분 강우.
[사건 창] 이벤트 시작 − 60분 ~ 끝 + 3시간. 기준 수위 = 시작 전 60분 관측값의 중앙값(관측 30분 이상일 때만).
  상승 = (수위 − 기준) × 1000 mm. 평활 상승 = 중심 10분 이동평균(창 안 관측 5개 이상일 때만, 아니면 결측).
[센서-사건 유효] 창 안 1분 관측률 ≥ 70%, 기준 수위 정의됨, 평활 상승 최댓값 ≥ 50 mm(상승 단계 1, docs/16).
[시각]
  T50 (주) = 평활 상승이 처음으로 그 사건 최댓값의 50% 이상이 된 시각("절반까지 오른 시각").
  Tpk (참고) = 평활 상승 최댓값 시각.
  강우 T50 = 창 안 누적 강우가 창 전체 강우의 50%에 처음 도달한 10분 블록의 끝 시각(창 강우 합 > 0 일 때만).
  반응 지연 D = 센서 T50 − 그 센서 강우량계의 강우 T50 (분).
[쌍] 센서 쌍(i<j, 순서 없음). 사건별 시차 lag = T50_j − T50_i, 강우 보정 시차 adj = D_j − D_i (= lag − 두 강우량계의 강우 T50 차이).
  두 센서가 모두 유효한 사건이 4개 이상인 쌍만 평가 쌍. 쌍 대표값 m = adj 의 사건 중앙값, 부호 일관성 c = sign(adj) == sign(m) 인 사건 비율
  (adj = 0 은 일치로 세지 않음, m = 0 이면 c = 0).
  "일관된 시차 쌍" = |m| ≥ 10분(10분 블록 하나 이상) 이고 c ≥ 0.75.
  도로 이웃 쌍 = 둘 중 하나가 다른 하나의 도로거리 최근접 5개 안(T1 G-road 그래프와 같은 기준, 같은 거리 행렬).
[판정 — 이 두 가지뿐. 나머지는 모두 참고]
  영가설(시차의 방향이 우연): 쌍마다 사건별 adj 의 부호를 독립으로 무작위로 뒤집는다(2000회, 시드 0). 매번 일관된 시차 쌍 비율을 다시 센다.
  L1 (주) 도로 이웃 쌍 중 일관된 시차 쌍 비율이 영가설 분포의 95백분위수보다 크면 "도로 이웃 사이 시차 신호 있음",
     아니면 "도로 이웃 사이 시차는 우연 수준과 구분 불가".
  L2 (보조) 전체 평가 쌍에서 같은 방식으로 "센서 쌍 시차 신호 있음" / "우연 수준과 구분 불가".
[참고 (판정에 쓰지 않음)]
  - 일관된 시차 쌍 중 |m| ≥ 30분(T1 예측 거리 이상)인 쌍의 수.
  - 센서별 반응 지연 D 의 사건 중앙값 분포.
  - |m| 과 도로거리의 순위 상관(Spearman), 이벤트 복원추출 2000회(시드 0) 95% 구간.
  - 강우 보정 전 lag 로 같은 집계(강우 도착 차이가 시차를 얼마나 만드는지).
  - Tpk 로 같은 집계.
[한계] 이벤트 13개. 신규 센서의 75~87분 주기 공백 때문에 유효 사건이 적은 센서가 있다. 강우는 10분 단위라 D 는 10분 오차를 가진다.
  센서 쌍끼리 겹치므로 쌍들은 독립이 아니다(영가설도 쌍 단위 독립 뒤집기라는 한계가 있다). 어느 쌍도 실제 관로 연결은 아니다.
==================================================================================================================================
"""
import numpy as np

FREEZE_DATE = "2026-11-14"
PRE_MIN = 60              # 기준 수위 구간(분)
POST_H = 3                # 이벤트 끝 이후(시간)
BASE_MIN_OBS = 30
SMOOTH = 10               # 중심 이동평균 길이(분)
SMOOTH_MIN_OBS = 5
COVER_MIN = 0.70
RISE_MIN_MM = 50.0
FRAC = 0.5
MIN_EVENTS = 4
LAG_MIN = 10.0            # 분
CONSIST_MIN = 0.75
LONG_LAG = 30.0           # 참고: T1 예측 거리
K_ROAD = 5
N_PERM = 2000
N_BOOT = 2000
SEED = 0


def smooth_rise(level_m: np.ndarray, base_m: float) -> np.ndarray:
    """1분 수위(m, NaN=결측) → 중심 SMOOTH분 이동평균 상승(mm). 창 안 관측이 SMOOTH_MIN_OBS 미만이면 NaN."""
    r = (level_m - base_m) * 1000.0
    ok = np.isfinite(r)
    v = np.where(ok, r, 0.0)
    k = np.ones(SMOOTH)
    s = np.convolve(v, k, mode="same")
    c = np.convolve(ok.astype(float), k, mode="same")
    return np.where(c >= SMOOTH_MIN_OBS, s / np.maximum(c, 1), np.nan)


def first_cross(x: np.ndarray, frac: float = FRAC) -> int | None:
    """x 가 처음으로 nanmax(x) × frac 이상이 되는 인덱스. 전부 NaN 이면 None."""
    if not np.isfinite(x).any():
        return None
    thr = np.nanmax(x) * frac
    idx = np.flatnonzero(np.isfinite(x) & (x >= thr))
    return int(idx[0])


def rain_t50_block(rain_10min: np.ndarray) -> int | None:
    """창 안 10분 강우 → 누적이 전체의 50%에 처음 도달한 블록 인덱스. 합이 0 이하면 None."""
    r = np.nan_to_num(rain_10min)
    tot = r.sum()
    if tot <= 0:
        return None
    return int(np.flatnonzero(np.cumsum(r) >= FRAC * tot)[0])


def pair_summary(adj: np.ndarray) -> tuple[float, float]:
    """사건별 시차 → (중앙값 m, 부호 일관성 c)."""
    m = float(np.median(adj))
    if m == 0:
        return m, 0.0
    return m, float(np.mean(np.sign(adj) == np.sign(m)))


def is_consistent(m: float, c: float) -> bool:
    return abs(m) >= LAG_MIN and c >= CONSIST_MIN


def consistent_fraction(pairs: list[np.ndarray]) -> float:
    if not pairs:
        return float("nan")
    return float(np.mean([is_consistent(*pair_summary(a)) for a in pairs]))


def sign_flip_null(pairs: list[np.ndarray], n_perm: int = N_PERM, seed: int = SEED) -> np.ndarray:
    """쌍마다 사건별 시차 부호를 독립으로 무작위 뒤집었을 때 일관된 시차 쌍 비율의 분포 (n_perm,)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for p in range(n_perm):
        out[p] = consistent_fraction([a * rng.choice((-1.0, 1.0), size=len(a)) for a in pairs])
    return out


def lag_verdict(observed: float, null: np.ndarray, yes: str, no: str) -> str:
    return yes if observed > np.percentile(null, 95) else no


L1_YES, L1_NO = "도로 이웃 사이 시차 신호 있음", "도로 이웃 사이 시차는 우연 수준과 구분 불가"
L2_YES, L2_NO = "센서 쌍 시차 신호 있음", "우연 수준과 구분 불가"
