"""T1 예측으로 '큰 상승(200 mm 이상) 첫 도달'을 30분 전에 잡는지 보는 사전 고정 규칙과 순수 함수. 실행 코드는 t1_warning.py.

==================================================================================================================================
사전 고정 규칙 (2026-11-14, T1 예측(gangnam2026_t1_st_predictions.npz)이 만들어지기 전이며 t1_st_eval 판정도 보기 전에 정함.
이후 변경 금지. 바꾸려면 새 파일로 만들고 "사후"로 표시한다)
==================================================================================================================================
[목적] T1 판정(docs/17)은 평균 오차(이벤트 MAE)다. 대시보드에서 쓰는 질문은 "30분 뒤 큰 상승이 오는지 미리 알 수 있나"이다.
  같은 T1 예측으로 상승 단계(docs/16) 기준의 경보 성능을 따로 본다. T1 판정 F1~F5 는 바꾸지 않는다.
[데이터] gangnam2026_t1_st_predictions.npz (모델 6개, h = 10·30분, 목표 시각 τ 기준 수위 예측), 평가 이벤트 13개(창 = 시작 ~ 끝 + 3시간,
  t2_loso.event_blocks), 예측 대상 17개.
[상승폭] 상승 = 수위 − 센서별 요일x시 중앙값(전체 기간, rise_levels.rise_mm 과 같은 정의) mm. 실제와 예측에 같은 평소 수위를 뺀다.
  유효 표본 = T1 판정과 같음(τ 실제와 M0 예측(= τ−h 실제)이 모두 있음).
[사례] (센서, 이벤트)마다 창 안 유효 표본 블록만 본다(유효 6개 미만이면 제외).
  도달 사례: 실제 상승이 처음 THRESH 이상이 되는 블록 τ* 가 있고, τ* 가 창의 첫 유효 블록이 아니며, 기점 τ*−h 의 실제 상승 < THRESH.
     (τ*−h 에 이미 THRESH 이상이었던 사례, 즉 '처음 도달'이 아닌 사례는 제외)
     맞힘 = 그 모델의 τ* 예측 상승 ≥ THRESH.
  비도달 사례: 창 안 실제 상승이 모두 THRESH 미만. 헛경보 = 창 안 어느 유효 블록에서든 예측 상승 ≥ THRESH.
  처음 도달하고 기점에 이미 넘었던 사례 등 위 둘에 들지 않는 사례는 세지 않는다.
[점수] CSI = 맞힘 수 / (도달 사례 수 + 헛경보 수). 높을수록 좋다. (M0 '직전 값 그대로'는 정의상 맞힘 0)
[판정 — 이 두 가지뿐] h = 30분, THRESH = 200 mm('큰 상승'). 이벤트 13개 2000회 복원추출(시드 0), 이벤트별 맞힘·도달·헛경보 합으로 CSI.
  W1 (주) CSI(G-road) − CSI(M2) 95% CI: 하한 > 0 → "GNN이 큰 상승 첫 도달을 더 잘 잡음" / 0 포함 → "구분 불가" /
     상한 < 0 → "GNN이 강우+자기이력 선형보다 못 잡음".
  W2 (주) CSI(G-corr) − CSI(M2), W1과 같은 3단계.
  GNN 그래프 2개를 비교했다는 사실을 함께 보고한다(다중 비교 보정 없음).
[참고 (판정에 쓰지 않음)] 모델별 맞힘률(맞힘/도달)·헛경보율(헛경보/비도달)·CSI, M3 − M2, h = 10분, THRESH = 500 mm('매우 큰 상승').
[한계] 평소 수위를 전체 기간으로 계산했다(실시간에서는 과거로만 계산해야 함, docs/16과 같은 한계). 도달 사례 수가 적다.
  상승 단계는 위험 판정이 아니다(관로 크기 미상).
==================================================================================================================================
"""
import numpy as np

FREEZE_DATE = "2026-11-14"
MAIN_H = 3
THRESH = 200.0
REF_THRESH = 500.0
MIN_VALID = 6
N_BOOT = 2000
SEED = 0
W_YES, W_ZERO, W_NO = "GNN이 큰 상승 첫 도달을 더 잘 잡음", "구분 불가", "GNN이 강우+자기이력 선형보다 못 잡음"


def case_outcome(actual: np.ndarray, pred: np.ndarray, issue_actual: np.ndarray, thresh: float = THRESH) -> tuple[str, bool] | None:
    """한 (센서, 이벤트)의 유효 블록 시계열(실제 상승, 예측 상승, 기점 실제 상승) → ("reach", 맞힘) / ("none", 헛경보) / None."""
    if len(actual) < MIN_VALID:
        return None
    above = np.flatnonzero(actual >= thresh)
    if len(above) == 0:
        return "none", bool((pred >= thresh).any())
    k = int(above[0])
    if k == 0 or issue_actual[k] >= thresh:
        return None
    return "reach", bool(pred[k] >= thresh)


def csi(hits: float, reach: float, false_alarms: float) -> float:
    den = reach + false_alarms
    return hits / den if den > 0 else float("nan")


def w_verdict(lo: float, hi: float) -> str:
    return W_YES if lo > 0 else (W_NO if hi < 0 else W_ZERO)
