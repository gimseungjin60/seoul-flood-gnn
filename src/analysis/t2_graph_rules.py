"""2026 그래프 보완 실험(고도 방향 그래프 G1, 랜덤 대조군 G2)의 사전 고정 규칙, 그래프 정의, 판정 함수.

==================================================================================================================================
사전 고정 규칙 (2026-11-09, 센서 고도를 계산하기 전이며 G1·G2로 어떤 모델도 돌리기 전에 정함. 이후 변경 금지.
바꾸려면 새 파일로 별도 실험을 만들고 "사후"로 표시한다)
==================================================================================================================================
[목적] 2026 재비교(docs/13)에서 도로거리 그래프 GNN은 IDW와 구분되지 않았다(P1). 배수분구 소속은 받을 수 없다는 답을 받았다(2026-11-09).
  관측 없이 정의되는 정보로 그래프를 바꿨을 때 달라지는지(G1), 지금 도로거리 그래프의 구조가 의미가 있는지(G2)를 묻는다.
  비교하는 새 그래프는 이 둘뿐이다(docs/12 "최대 2개"). docs/13의 판정(P1~P3, S1)은 바꾸지 않는다.
[고정하는 것] 2026 재비교와 전부 같다(src/analysis/t2_2026_rules.py, docs/11): 데이터·품질 규칙, 센서 역할(타깃 17, pool 48), 5폴드·purge,
  평가 이벤트 13개, GNN 하이퍼파라미터(k=5, 은닉 32, 2층, 30에폭, 시드 0·1·2 평균), 가짜 타깃·손실 = core(labels), 메시지 가중치 1/도로거리,
  관측된 이웃만 가중평균. **바뀌는 것은 각 노드가 메시지를 받는 이웃 집합뿐이다.** 비교 기준(도로거리 GNN, IDW)은 이미 계산된 2026 예측을 그대로 쓴다.
[센서 고도] Copernicus DEM GLO-30(30 m, 타일 N37 E127). 센서 좌표(gangnam2026_station_nodes.csv)가 들어 있는 화소를 중심으로 3x3 화소 중 최솟값.
  GLO-30은 건물·수목이 포함된 표면 고도(DSM)라서, 도로(주변보다 낮음) 위 값을 고르려고 최솟값을 쓴다. 고도는 위치로 정해지는 값이며 관측이 아니다.
[G1 고도 방향 그래프] 받는 노드 r 마다 지금과 같은 도로거리 최근접 5개(같은 후보 집합)를 고른 뒤, 그중 고도가 r 보다 높은 이웃(elev > elev_r)만 남긴다.
  물이 높은 곳에서 낮은 곳으로 흐르므로, 낮은 센서는 위쪽 센서에서만 정보를 받는다. 남는 이웃이 없으면 5개를 그대로 쓴다(대체).
  학습(타깃 제외 노드끼리)과 평가(타깃 행) 모두 같은 규칙. 대체가 일어나는 노드 수는 모델 실행 전에 보고한다.
[G2 랜덤 대조군] 그래프 시드 g = 0, 1, 2. 시드마다 pool 48개의 (받는 노드, 보내는 노드) 쌍에 균등난수 점수를 한 번 뽑고(np.random.default_rng(RANDOM_GRAPH_SEED_BASE + g)),
  받는 노드마다 후보 중 점수가 가장 작은 5개를 이웃으로 한다(이웃 수는 도로거리 그래프와 같음, 가중치는 그 쌍의 1/도로거리).
  그래프 시드 g 는 GNN 시드 g 와 짝을 이룬다. G2 예측 = 세 (그래프, 모델) 쌍 예측의 평균(도로거리 GNN의 시드 3개 평균과 같은 계산량).
[판정 — 이 세 가지뿐. 나머지는 모두 참고]
  부트스트랩: 2026 재비교와 같다(평가 이벤트 13개 2000회 복원추출, 시드 0, 타깃 17개 평균, 차이 A − B 의 95% CI).
  Q1 (G1 주) G1 GNN − 도로거리 GNN 이벤트 MAE. 상한 < 0 → "고도 방향 그래프가 나음" / 0 포함 → "구분 불가" / 하한 > 0 → "고도 방향 그래프가 나쁨".
  Q2 (G1 보조) G1 GNN − IDW 이벤트 MAE. 상한 < 0 → "G1 GNN이 IDW를 넘음" / 0 포함 → "구분 불가" / 하한 > 0 → "IDW보다 나쁨".
  Q3 (G2) 도로거리 GNN − G2 GNN 이벤트 MAE. 상한 < 0 → "도로거리 구조가 랜덤보다 나음" / 0 포함 → "도로거리 구조의 기여 확인 안 됨"
     / 하한 > 0 → "랜덤보다 나쁨".
  새 그래프 2개를 비교했다는 사실을 결과와 함께 보고한다(다중 비교 보정은 하지 않음).
[참고, 판정 아님] G2 GNN − IDW, 센서별 MAE, 대체 노드 수, 그래프 시드별 G2 결과.
[한계] DSM 30 m 고도는 실제 관로 매설 깊이·경사가 아니다. 지표 고도 차가 작은 평지에서는 방향이 잡음일 수 있다. 실제 관로 연결이 아니라는 점은 같다.
==================================================================================================================================
"""
import numpy as np

from src.analysis.t2_2026_rules import three_way

FREEZE_DATE = "2026-11-09"
GRAPHS = ("elev", "rand")                          # G1, G2 (최대 2개)
K = 5                                              # t2_gnn.K_NEIGHBORS 와 같음
DEM_TILE = "Copernicus_DSM_COG_10_N37_00_E127_00_DEM"
DEM_URL = f"https://copernicus-dem-30m.s3.amazonaws.com/{DEM_TILE}/{DEM_TILE}.tif"
DEM_WINDOW = 3                                     # 3x3 화소 최솟값
RANDOM_GRAPH_SEED_BASE = 1000


def nearest(dist: np.ndarray, r: int, cols: list, k: int = K) -> list:
    """cols(자기 자신 제외) 중 r 에서 도로거리가 가장 가까운 k개의 cols 내 위치. t2_gnn.knn_weights 와 같은 순서."""
    return [b for _, b in sorted((dist[r, c], b) for b, c in enumerate(cols) if c != r)[:k]]


def elev_neighbors(dist: np.ndarray, elev: np.ndarray, r: int, cols: list, k: int = K) -> tuple[list, bool]:
    """G1: 도로거리 최근접 k개 중 r 보다 높은 이웃만. 반환 (cols 내 위치 목록, 대체 여부)."""
    near = nearest(dist, r, cols, k)
    up = [b for b in near if elev[cols[b]] > elev[r]]
    return (up, False) if up else (near, True)


def rand_neighbors(score: np.ndarray, r: int, cols: list, k: int = K) -> list:
    """G2: 점수가 가장 작은 k개(자기 제외)."""
    return [b for _, b in sorted((score[r, c], b) for b, c in enumerate(cols) if c != r)[:k]]


def random_scores(n: int, g: int) -> np.ndarray:
    return np.random.default_rng(RANDOM_GRAPH_SEED_BASE + g).random((n, n))


def weights_from(dist: np.ndarray, rows: list, cols: list, pick) -> np.ndarray:
    """rows 각 노드가 pick(r) 이 고른 cols 내 위치에서 받는 1/d 가중치 (len(rows), len(cols))."""
    W = np.zeros((len(rows), len(cols)))
    for a, r in enumerate(rows):
        for b in pick(r):
            W[a, b] = 1.0 / dist[r, cols[b]]
    return W


def q1_verdict(lo: float, hi: float) -> str:
    return three_way(lo, hi, "고도 방향 그래프가 나음", "고도 방향 그래프가 나쁨")


def q2_verdict(lo: float, hi: float) -> str:
    return three_way(lo, hi, "G1 GNN이 IDW를 넘음", "IDW보다 나쁨")


def q3_verdict(lo: float, hi: float) -> str:
    v = three_way(lo, hi, "도로거리 구조가 랜덤보다 나음", "랜덤보다 나쁨")
    return "도로거리 구조의 기여 확인 안 됨" if v == "구분 불가" else v
