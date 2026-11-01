"""2026 우기 재비교의 사전 고정 규칙과 판정 함수. 모델 실행 코드는 없다(실행 코드는 이 모듈의 상수·함수만 가져다 쓴다).

==================================================================================================================================
사전 고정 규칙 (2026-11-01, 2026 수위로 어떤 모델도 돌리기 전에 정함. 이후 변경 금지. 바꾸려면 새 파일로 별도 실험을 만들고 사후로 표시한다)
==================================================================================================================================
[목적] 2025 결과(T2 GNN − IDW "구분 불가", docs/10)를 센서가 더 많은 2026 우기에서 다시 확인한다. 새 설정 탐색은 하지 않는다.
[데이터] data/processed/gangnam2026_station_timeseries.npz (품질 규칙 R1·R2 적용, src/data/prepare_station_timeseries_2026.py).
  2026-06-01~08-31 전체를 시간순 연속 5폴드 교차검증, 평가 폴드 양쪽 24시간 purge (2025와 같음). 10분 블록(관측 5분 이상).
  2026 자료로는 하이퍼파라미터를 고르지 않으므로 별도 holdout을 두지 않는다. 2025 test 구간(08-18~31)은 계속 읽지 않는다.
[센서]
  타깃(평가 대상) = core(1분 커버리지 >= 0.90) 20개 중 좌표가 확인된 센서.
  가짜 타깃(학습 라벨) = 타깃을 뺀 나머지 core. neighbor 센서(10분 블록 >= 0.50)는 이웃 입력으로만 쓴다(라벨 아님).
  이웃 후보 = pool(core + neighbor) − 현재 타깃 − 아래 제외 센서.
  제외 규칙 J1: R1·R2 적용 후 연속 1분 |Δ수위| >= 1 m 횟수가 100회 이상인 센서는 pool에서 뺀다(현재 23-0052, 1,811회).
  제외 규칙 J2: 좌표를 확보하지 못했거나 좌표 검증을 통과하지 못한 센서는 pool에서 뺀다(타깃이면 타깃에서도 빼고 보고).
[신규 센서 검증 H] H = 2025 분석 18개(gangnam_station_graph_nodes.csv)에 없던 core 센서
  = 23-0011, 23-0018(2025 좌표 미확보), 23-0016(2025-08-29 측정 시작), 23-0039, 23-0041, 23-0066 (6개).
  주 분석과 별도로, H 전체를 학습(가짜 타깃, 이웃 입력, 표준화 통계)에서 빼고 나머지로 학습한 모델로 H를 추정한다.
  센서 6개뿐이라 판정이 아니라 참고로 보고한다(지표·CI 방식은 주 판정과 같음).
[이벤트] 2025와 같은 규칙(강남 강우량계 3개소 평균, 6시간 무강우로 분리, 총량 >= 10 mm). 2026은 14개.
  자료 끝(08-31 23:59)에서 잘린 이벤트(끝 + 3시간이 자료 끝을 넘는 이벤트)는 평가에서 뺀다 → 13개(50 mm 이상 5개 포함).
  이벤트 창 = 시작 ~ 끝 + 3시간.
[모델·하이퍼파라미터] 2025 사전 고정값 그대로(src/analysis/t2_loso.py, t2_gnn.py 상수). 그리드 탐색 없음. GNN 시드 0·1·2 평균.
  IDW는 그 블록에서 관측된 이웃 후보 전체(p=1, 도로거리). GNN 그래프는 이웃 후보 중 도로거리 최근접 k=5(관측된 이웃만 가중평균).
  GNN 이웃은 관측 여부와 무관하게 거리로 고정한다(2025와 같음). 신규 센서가 주기적으로 비어 GNN의 유효 이웃이 IDW보다 적을 수 있다.
  이 차이를 줄이려 그래프를 블록마다 바꾸는 것은 새 모델이므로 하지 않는다. 대신 모델 실행 전에 타깃별 k=5 이웃의 블록 관측률을 계산해 보고한다.
[지표] 이벤트 MAE(주), 첨두 오차(mm), 첨두 시각 오차(분). 첨두 두 지표의 정의는 src/analysis/t2_gnn_peaks.py와 같다
  (이벤트 창 안에서 정답·이웃 정보가 있는 블록이 6개 이상인 센서만, |예측 최대 − 실제 최대|, |예측 최대 시각 − 실제 최대 시각|).
[판정 — 이 네 가지뿐. 나머지는 모두 참고]
  부트스트랩: 평가 이벤트 13개를 2000회 복원추출(시드 0), 타깃 센서 평균(매크로). 차이 A − B 의 95% CI(2.5/97.5 백분위).
  P1 (주)  GNN − IDW 이벤트 MAE.        상한 < 0 → "GNN이 IDW를 넘음" / 0 포함 → "구분 불가" / 하한 > 0 → "IDW보다 나쁨".
  P2 (보조) GNN − (4) 강우+이웃(선형) 이벤트 MAE. P1이 "넘음"이고 이 CI 상한도 < 0 일 때만 "그래프 구조 효과"로 보고한다.
  P3 (첨두 시각, 새 사전 지표) GNN − IDW 첨두 시각 오차(분). P1과 같은 3단계. P1의 결론을 바꾸지 않고 따로 보고한다.
     대상: 실제 첨두 상승(이벤트 창 안 실제 상승분의 최대)이 50 mm 이상인 (센서, 이벤트) 조합만(측정 단위 10 mm의 5배).
     근거(2025 IDW, 사후 결과로 확인): 상승 50 mm 미만 조합(144개 중 14개)은 시각 오차 평균 약 200분으로 첨두가 잡음 수준이었다.
     통계량: 한 반복에서 센서별로 뽑힌 이벤트의 대상 조합 시각 오차 평균 → 대상 조합이 하나라도 있는 센서들의 평균.
  S1 (공간 신호 재현) (4) 또는 (5)가 (3) 강우+시간대 대비 이벤트 MAE를 10% 이상 줄이는 타깃이 타깃 수의 1/3 이상
     (2025의 18개 중 6개와 같은 비율, 타깃 20개면 7개) → "T2 공간 신호 있음".
  2025 결과와의 비교는 서술로만 하고 판정에 쓰지 않는다.
[한계] 이벤트 13개(2025는 8개). 신규 센서 다수가 2시간 주기로 75~87분씩 비어 있어, 이웃 입력이 블록마다 달라진다.
  그래프는 도로거리 근사이며 실제 관로 연결이 아니다. 타깃의 기저선·규모를 안다는 가정(상대 상승폭 추정)은 2025와 같다.
==================================================================================================================================
"""
import math

import numpy as np
import pandas as pd

FREEZE_DATE = "2026-11-01"
N_FOLDS, PURGE_BLOCKS = 5, 144
EVENT_AFTER_H = 3
N_BOOT, BOOT_SEED = 2000, 0
JUMP_M, MAX_JUMPS = 1.0, 100                       # J1
NEW_SENSORS_H = ("23-0011", "23-0016", "23-0018", "23-0039", "23-0041", "23-0066")  # H (core 중 2025 18개 밖)
GNN_K = 5
PEAK_MIN_BLOCKS = 6
P3_MIN_RISE_M = 0.05                               # P3 대상: 실제 첨두 상승 >= 50 mm
S1_IMPROVE, S1_SHARE = 0.10, 1 / 3


def three_way(lo: float, hi: float, better: str, worse: str) -> str:
    """차이 A − B 의 95% CI (음수가 A가 좋은 쪽)."""
    if hi < 0:
        return better
    if lo > 0:
        return worse
    return "구분 불가"


def p1_verdict(lo: float, hi: float) -> str:
    return three_way(lo, hi, "GNN이 IDW를 넘음", "IDW보다 나쁨")


def p2_graph_effect(p1: str, hi_vs4: float) -> bool:
    return p1 == "GNN이 IDW를 넘음" and hi_vs4 < 0


def p3_verdict(lo: float, hi: float) -> str:
    return three_way(lo, hi, "첨두 시각에서 GNN이 IDW를 넘음", "첨두 시각에서 IDW보다 나쁨")


def p3_eligible(actual_window_m: np.ndarray) -> bool:
    """P3 대상 (센서, 이벤트) 조합인지: 이벤트 창 안 유효 블록의 실제 상승분 최대가 50 mm 이상."""
    return bool(np.nanmax(actual_window_m) >= P3_MIN_RISE_M - 1e-9)


def s1_min_sensors(n_targets: int) -> int:
    return math.ceil(n_targets * S1_SHARE - 1e-9)


def s1_verdict(n_improved: int, n_targets: int) -> bool:
    return n_improved >= s1_min_sensors(n_targets)


def eval_events(events: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """유의 이벤트 중 창(끝 + 3시간)이 자료 끝 안에 들어오는 것만."""
    keep = events["end"] + pd.Timedelta(hours=EVENT_AFTER_H) <= data_end
    return events[keep].reset_index(drop=True)


def jump_excluded(values: np.ndarray, node_ids) -> list[str]:
    """J1: 연속 1분 |Δ| >= 1 m 횟수가 100회 이상인 센서 (NaN 사이는 세지 않음)."""
    jumps = (np.abs(np.diff(values.astype(np.float64), axis=0)) >= JUMP_M).sum(axis=0)
    return [str(n) for n, c in zip(node_ids, jumps) if c >= MAX_JUMPS]


def sensor_roles(sel: pd.DataFrame, jump_ex: list[str], no_coord: list[str]) -> dict:
    """선정표(gangnam2026_sensor_selection.csv)와 J1·J2 제외 목록으로 역할을 정한다."""
    drop = set(jump_ex) | set(no_coord)
    pool = [n for n in sel.loc[sel["in_pool"], "node_id"] if n not in drop]
    core = [n for n in sel.loc[sel["is_core"], "node_id"] if n in pool]
    return {"targets": core, "pool": pool, "H": [n for n in NEW_SENSORS_H if n in core],
            "dropped": sorted(drop & set(sel.loc[sel["in_pool"], "node_id"]))}
