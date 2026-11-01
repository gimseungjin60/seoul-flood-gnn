"""T2 LOSO 위에 GNN 추가: 센서 하나를 통째로 빼고, 이웃 센서의 '현재' 상승분과 타깃 지점 강우·시간대로 그 센서의 기저 대비 상승분을 추정한다.
기존 T2 LOSO(src/analysis/t2_loso.py)와 같은 평가 틀(18개 타깃, 시간순 5폴드, purge 24h, 10분 블록, 이벤트 8개, test 미사용)을 쓴다.

실행: python -m src.analysis.t2_gnn --timing      # 타깃 1 x 폴드 1 x 시드 1 실행 시간만 측정(결과 저장 안 함)
      python -m src.analysis.t2_gnn               # 전체 270회(18 타깃 x 5 폴드 x 3 시드) 실행, 예측 저장
      python -m src.analysis.t2_gnn_eval          # 사전 기준에 따른 판정 (먼저 위 실행이 끝나야 한다)

==================================================================================================================================
사전 고정 규칙 (2026-10-18, 결과를 보기 전에 정함. 이후 변경 금지. 바꾸려면 새 파일로 별도 실험을 만들고 사후로 표시한다)
==================================================================================================================================
[하이퍼파라미터 — 그리드 탐색 없음]
  k=5, 마스크 비율 p=0.3, 이벤트 가중 w=5, 은닉 32, 층 2, 학습률 1e-3(Adam, weight decay 1e-4), 에폭 30, 배치 256, Huber(delta=1.0).
  에폭은 고정이며 val 기반 조기 종료는 없다. 시드 3개(0,1,2)의 예측을 평균한 것이 GNN 예측이다.
  그리드(k∈{3,5}, p∈{0.2,0.4}, w∈{1,5})는 판정 후 민감도 분석으로만 돌리며 "사후, 판정에 쓰지 않음"으로 표시한다(판정 보고서가 있어야 실행됨).

[설정]
  바깥 루프: 타깃 센서 1개를 통째로 제외(18번). 안쪽: 나머지 17개 중 일부(p)를 무작위로 가리고 가려진 노드의 정규화 상승분을 맞히도록 학습(IGNNK 계열).
  타깃의 과거값은 어떤 형태로도 입력 금지: 타깃 열은 loso 함수 맨 앞에서 제거되고, 평가 입력의 타깃 노드는 (상승분 0, 관측 플래그 0)이다.
  타깃의 좌표(도로거리)와 최근접 강우량계는 관측이 아니므로 사용한다. 타깃의 기저선·규모는 라벨 정의와 단위 환산에만 쓰고 입력에는 넣지 않는다(기존과 동일한 가정:
  '이력이 있던 센서를 가린 경우'의 상대 상승폭 추정이며 절대수위 추정이 아니다).
[그래프] 도로거리 기반만. 각 노드는 가장 가까운 k개 노드(자기 제외, 학습 시에는 17개 중, 평가 시 타깃은 17개 중)에서 메시지를 받는다. 가중치 1/d, 가려진·결측 이웃은 제외하고
  정규화(관측된 이웃만의 거리가중 평균). 엣지는 관측 없이 정의된다(같은 구는 후보가 아님: 센서가 모두 강남구).
[입력] 노드별: 현재 정규화 상승분(가려지면 0), 관측 플래그, 강우 특징 10개(최근 6슬롯 log1p, 결측 수, 선행 누적 3h/6h/24h log1p; 그 노드의 최근접 강우량계), 시간대 3개(sin/cos, 주말).
  특징 표준화 통계는 학습 구간의 17개 노드로만 계산한다.
[모델] 선형 인코더(32) → 메시지 전달 2층(h' = ReLU(W_self h + W_nbr · 관측 이웃 가중평균(h))) → 선형 출력(1). dense 인접행렬, 그래프 라이브러리 미사용.
[학습 손실] 가려진 노드(+정답이 있는 노드)만 Huber, 이벤트 구간 블록(시작~끝+3h)에는 가중 w. 가림은 매 반복 새로 뽑는다(전 노드가 가려지는 표본은 손실에서 제외).
[판정 — 코드에 먼저 고정] (평가 이벤트 8개를 2000회 복원추출, 시드 0, 센서 18개 평균 이벤트 MAE, 기존 t2_loso_bootstrap 과 같은 방식)
  주 판정: GNN − IDW 차이의 95% CI.  상한 < 0 → "GNN이 IDW를 넘음" / 0 포함 → "구분 불가" / 하한 > 0 → "IDW보다 나쁨".
  보조 판정: GNN − (4) 강우+이웃(선형) 차이의 95% CI.  주 판정이 "GNN이 IDW를 넘음"이고 이 CI의 상한도 < 0 일 때만 "그래프 구조 효과"로 보고한다.
  참고(판정 아님): val 이벤트(7·8번) 결과, 큰 사건 제외 등. 이 이벤트들은 하이퍼파라미터 선택에 쓰이지 않았다(하이퍼파라미터는 위에서 사전에 고정).
[데이터] train+val 구간(분 인덱스 < 112,608)만. test 는 읽지 않는다.
[한계] 이벤트 8개뿐이라 CI가 넓다. 학습 풀이 17개 센서뿐이다. 선형(4)·트리(5)가 IDW와 구분되지 않았던 선행 결과(docs/08)를 감안하면 GNN도 구분 불가일 수 있다.
==================================================================================================================================
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.analysis.t2_loso import (EVENT_AFTER_H, N_FOLDS, build_inputs, event_blocks, fold_slices, neighbor_features,
                                  normalized_rise)
from src.data.clean_nodes import PROCESSED

# ----------------------------------------------------------------------------- 사전 고정 하이퍼파라미터 (변경 금지)
K_NEIGHBORS = 5
MASK_RATIO = 0.3
EVENT_WEIGHT = 5.0
HIDDEN = 32
N_LAYERS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
BATCH = 256
HUBER_DELTA = 1.0
SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = 2000, 0
SD_FLOOR = 1e-6

OUT_PRED = PROCESSED / "gangnam_t2_gnn_predictions.npz"   # 재생성 가능, git 제외 대상
OUT_REPORT = PROCESSED / "gangnam_t2_gnn_report.txt"
OUT_CI = PROCESSED / "gangnam_t2_gnn_ci.csv"


# ----------------------------------------------------------------------------- 그래프
def knn_weights(dist: np.ndarray, rows: list, cols: list, k: int = K_NEIGHBORS) -> np.ndarray:
    """rows 각 노드가 cols(자기 자신 제외) 중 가장 가까운 k개에서 받는 1/d 가중치. 반환 (len(rows), len(cols)), 나머지는 0."""
    W = np.zeros((len(rows), len(cols)))
    for a, r in enumerate(rows):
        cand = sorted((dist[r, c], b) for b, c in enumerate(cols) if c != r)
        for d, b in cand[:k]:
            W[a, b] = 1.0 / d
    return W


def build_adjacency(dist: np.ndarray, others: list, target: int | None) -> np.ndarray:
    """학습용(target=None): others 17개 노드끼리. 평가용: others 뒤에 타깃 노드를 하나 더 붙인 (n+1, n+1) 행렬.
    타깃 열은 항상 0이라 어떤 노드도 타깃에서 메시지를 받지 않는다(타깃은 관측이 없는 노드)."""
    n = len(others)
    size = n + (1 if target is not None else 0)
    A = np.zeros((size, size))
    A[:n, :n] = knn_weights(dist, others, others)
    if target is not None:
        A[n, :n] = knn_weights(dist, [target], others)[0]
    return A


# ----------------------------------------------------------------------------- 모델
class MaskAwareGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, n_layers: int = N_LAYERS):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.self_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.nbr_lin = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x (B,N,F), m (B,N) 관측 플래그(0/1), A (N,N) 메시지 가중치(행=받는 노드). 반환 (B,N) 정규화 상승분 추정."""
        h = torch.relu(self.enc(x))
        w = A[None] * m[:, None, :]                                   # 관측된 이웃만
        den = w.sum(-1, keepdim=True).clamp_min(1e-9)
        w = w / den                                                   # 관측 이웃이 없으면 w=0 → 메시지 0
        for sl, nl in zip(self.self_lin, self.nbr_lin):
            h = torch.relu(sl(h) + nl(w @ h))
        return self.out(h).squeeze(-1)


# ----------------------------------------------------------------------------- 입력 구성 (타깃 열 제거 후에만 호출된다)
def feature_stats(static: np.ndarray, train_idx: np.ndarray):
    """static (B, N, 13): 강우 10 + 시간대 3. 학습 구간·학습 노드로만 평균/표준편차."""
    x = static[train_idx].reshape(-1, static.shape[-1])
    mu = x.mean(0)
    sd = np.where(x.std(0) > SD_FLOOR, x.std(0), 1.0)
    return mu, sd


def event_weight_vector(bt: pd.DatetimeIndex, events: pd.DataFrame, w: float = EVENT_WEIGHT) -> np.ndarray:
    v = np.ones(len(bt))
    for _, idx in event_blocks(bt, events):
        v[idx] = w
    return v


def make_eval_inputs(R_oth: np.ndarray, static_oth: np.ndarray, static_target: np.ndarray, eval_idx: np.ndarray, mu, sd):
    """평가 입력: (17 이웃 + 타깃 노드). 타깃 노드의 상승분은 0, 관측 플래그는 0 (R 에 타깃 열이 없으므로 접근 자체가 불가능)."""
    B = len(eval_idx)
    Rn = R_oth[eval_idx]
    ok = np.isfinite(Rn)
    stat = np.concatenate([static_oth[eval_idx], static_target[eval_idx][:, None, :]], axis=1)
    stat = (stat - mu) / sd
    r = np.concatenate([np.where(ok, Rn, 0.0), np.zeros((B, 1))], axis=1)
    m = np.concatenate([ok.astype(float), np.zeros((B, 1))], axis=1)
    x = np.concatenate([r[..., None], m[..., None], stat], axis=2)
    return x.astype(np.float32), m.astype(np.float32)


def train_model(R_oth: np.ndarray, static_oth: np.ndarray, train_idx: np.ndarray, A17: np.ndarray, weights: np.ndarray, seed: int,
                mu, sd, epochs: int = EPOCHS, label_mask: np.ndarray | None = None):
    """17개 센서(타깃 제외)로 가림-맞히기 학습. 반환 모델.
    label_mask (N_oth,) 1=손실 대상 노드(2026: core). None 이면 전 노드(2025와 같음). 가림은 전 노드에 똑같이 적용한다."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    in_dim = 2 + static_oth.shape[-1]
    model = MaskAwareGNN(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Rt = R_oth[train_idx]
    obs = np.isfinite(Rt)
    stat = ((static_oth[train_idx] - mu) / sd).astype(np.float32)
    Rz = torch.tensor(np.where(obs, Rt, 0.0), dtype=torch.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    stat_t = torch.tensor(stat)
    w_t = torch.tensor(weights[train_idx], dtype=torch.float32)
    A = torch.tensor(A17, dtype=torch.float32)
    lab = torch.ones(Rz.shape[1]) if label_mask is None else torch.tensor(label_mask, dtype=torch.float32)
    n = len(train_idx)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, BATCH):
            b = perm[s: s + BATCH]
            drop = (torch.rand(len(b), Rz.shape[1], generator=gen) < MASK_RATIO).float()
            m = obs_t[b] * (1 - drop)
            tgt = obs_t[b] * drop * lab[None, :]                          # 가려졌고 정답이 있는 (라벨) 노드
            x = torch.cat([(Rz[b] * m)[..., None], m[..., None], stat_t[b]], dim=2)
            pred = model(x, m, A)
            hub = nn.functional.huber_loss(pred, Rz[b], reduction="none", delta=HUBER_DELTA)
            ww = w_t[b][:, None] * tgt
            if float(ww.sum()) <= 0:
                continue
            loss = (hub * ww).sum() / ww.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def prepare_target(R: np.ndarray, rain_feat: list, time_feat: np.ndarray, i: int) -> dict:
    """타깃 열을 가장 먼저 제거하고, 이후 단계가 쓰는 입력만 만든다. 반환 dict 에는 타깃의 수위 시계열이 어디에도 없다."""
    N = R.shape[1]
    others = [j for j in range(N) if j != i]
    R_oth = np.delete(R, i, axis=1)                                       # 타깃 열 제거
    static_all = np.stack([np.hstack([rain_feat[j], time_feat]) for j in range(N)], axis=1)   # (B, N, 13): 위치로 정해지는 입력(관측 아님)
    return dict(others=others, R_oth=R_oth, static_oth=static_all[:, others], static_tgt=static_all[:, i])


def gnn_loso_predict(R: np.ndarray, dist: np.ndarray, rain_feat: list, time_feat: np.ndarray, i: int, train_idx: np.ndarray,
                     eval_idx: np.ndarray, weights: np.ndarray, seeds=SEEDS, epochs: int = EPOCHS, labels=None) -> np.ndarray:
    """타깃 센서 i 를 뺀 GNN 예측(정규화 상승분, (len(eval_idx),)). 이웃이 하나도 관측되지 않은 블록은 NaN(IDW 등과 같은 정의).
    R (B, N) 의 i 열은 가장 먼저 제거되어 이후 어떤 계산에도 쓰이지 않는다(누수 방지 테스트가 검증).
    labels: 손실 대상 노드 인덱스(2026: core만). None 이면 i 를 뺀 전 노드(2025와 같음)."""
    c = prepare_target(R, rain_feat, time_feat, i)
    others, R_oth, static_oth = c["others"], c["R_oth"], c["static_oth"]
    label_mask = None if labels is None else np.array([j in set(labels) for j in others], dtype=float)
    mu, sd = feature_stats(static_oth, train_idx)
    A17 = build_adjacency(dist, others, None)
    A18 = build_adjacency(dist, others, i)
    x_e, m_e = make_eval_inputs(R_oth, static_oth, c["static_tgt"], eval_idx, mu, sd)
    preds = []
    for sd_ in seeds:
        model = train_model(R_oth, static_oth, train_idx, A17, weights, sd_, mu, sd, epochs, label_mask)
        with torch.no_grad():
            p = model(torch.tensor(x_e), torch.tensor(m_e), torch.tensor(A18, dtype=torch.float32))[:, -1].numpy()
        preds.append(p)
    out = np.mean(preds, axis=0).astype(float)
    _, nv = neighbor_features(R_oth[eval_idx], dist[i, others])           # 기존 모델과 같은 '이웃 정보 없음' 정의
    out[nv == 0] = np.nan
    return out


# ----------------------------------------------------------------------------- 실행
def run_gnn_loso(inp: dict, log=None, targets=None, folds=None, seeds=SEEDS, epochs: int = EPOCHS):
    blk, bt = inp["blk"], inp["bt"]
    B, N = blk.shape
    pred = np.full((B, N), np.nan)
    act = np.full((B, N), np.nan)
    wv = event_weight_vector(bt, inp["events"])
    for f, (eval_idx, train_idx, train_mask) in enumerate(fold_slices(B)):
        if folds is not None and f not in folds:
            continue
        R, scale = normalized_rise(blk, inp["how"], train_mask)
        t0 = time.time()
        for i in (range(N) if targets is None else targets):
            p = gnn_loso_predict(R, inp["dist"], inp["rain_feat"], inp["time_feat"], i, train_idx, eval_idx, wv, seeds, epochs)
            pred[eval_idx, i] = p * scale[i]
            act[eval_idx, i] = R[eval_idx, i] * scale[i]
        if log:
            log(f"  fold {f + 1}/{N_FOLDS} 완료 ({time.time() - t0:.0f}초)")
    return pred, act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true", help="타깃 1 x 폴드 1 x 시드 1 실행 시간만 측정")
    args = ap.parse_args()
    torch.set_num_threads(1)
    inp = build_inputs()
    log = lambda s: print(s, flush=True)
    B, N = inp["blk"].shape
    log(f"블록 {B:,}개, 센서 {N}개, 폴드 {N_FOLDS}, 시드 {len(SEEDS)}")
    if args.timing:
        t0 = time.time()
        run_gnn_loso(inp, log, targets=[0], folds=[0], seeds=(0,))
        dt = time.time() - t0
        total = dt * N * N_FOLDS * len(SEEDS)
        log(f"1회(타깃 1 x 폴드 1 x 시드 1, 에폭 {EPOCHS}) {dt:.1f}초 → 전체 {N * N_FOLDS * len(SEEDS)}회 예상 {total / 60:.0f}분 ({total / 3600:.2f}시간)")
        return
    t0 = time.time()
    pred, act = run_gnn_loso(inp, log)
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array(inp["nodes"]),
                        actual_m=act.astype(np.float32), pred_gnn=pred.astype(np.float32))
    log(f"저장: {OUT_PRED.name} (총 {time.time() - t0:.0f}초). 판정은 python -m src.analysis.t2_gnn_eval")


if __name__ == "__main__":
    main()
