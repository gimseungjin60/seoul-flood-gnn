"""T1 시공간 GNN 단기 예측 실행: 사전 고정 규칙(src/analysis/t1_st_rules.py, docs/17)대로 M0~M3 와 G-road·G-corr 예측을 만든다.
판정은 python -m src.analysis.t1_st_eval 에서만 한다.

실행: python -m src.analysis.t1_st --timing     # GNN 1회(폴드 1 x 시드 1 x 그래프 1) 시간만 측정
      python -m src.analysis.t1_st              # 전체 (5폴드 x 3시드 x 그래프 2 + Ridge)
출력: data/processed/gangnam2026_t1_st_predictions.npz (재생성 가능, git 제외)

예측은 '목표 시각' τ = t + h 기준으로 저장한다. τ 의 예측은 기점 t = τ − h 가 평가 폴드에 있을 때만 만든다.
기점 t 의 예측에는 t 이후 블록의 값이 들어가지 않는다(누수 테스트가 검증). 정규화·상관 그래프는 학습 구간만으로 만든다.
"""
import argparse
import time

import numpy as np
import torch
from torch import nn

from src.analysis.t1_st_rules import (BATCH, CORR_MIN_PAIRS, EPOCHS, EVENT_WEIGHT, HIDDEN, HORIZONS, HUBER_DELTA, K, LR, SEEDS,
                                      WEIGHT_DECAY, WINDOW)
from src.analysis.t2_2026 import build_inputs, load_roles
from src.analysis.t2_gnn import event_weight_vector, knn_weights
from src.analysis.t2_loso import N_FOLDS, RIDGE_LAMBDA, _ridge, fold_slices, normalized_rise
from src.data.clean_nodes import PROCESSED

OUT_PRED = PROCESSED / "gangnam2026_t1_st_predictions.npz"
SD_FLOOR = 1e-6


# ----------------------------------------------------------------------------- 입력 (기점 t 기준, t 이후 값은 쓰지 않음)
def history(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a (B, N) 정규화 수위(NaN=결측) → (B, N, WINDOW) 값(결측 0), 관측 플래그. 앞 WINDOW−1 블록은 창이 모자라 관측 0."""
    B, N = a.shape
    val = np.zeros((B, N, WINDOW))
    obs = np.zeros((B, N, WINDOW))
    for k in range(WINDOW):
        lag = WINDOW - 1 - k                           # k = WINDOW−1 이 t
        src = np.full((B, N), np.nan)
        src[lag:] = a[: B - lag]
        val[..., k] = np.nan_to_num(src)
        obs[..., k] = np.isfinite(src)
    return val, obs


def self_features(val: np.ndarray, obs: np.ndarray, j: int) -> np.ndarray:
    """M1 특징: a[t], a[t] − a[t−k] (k=1..5, 결측이면 0), 창 안 결측 수."""
    cur = val[:, j, -1]
    diffs = [np.where(obs[:, j, -1 - k] > 0, cur - val[:, j, -1 - k], 0.0) for k in range(1, WINDOW)]
    return np.column_stack([cur, *diffs, WINDOW - obs[:, j].sum(1)])


def neighbor_features(val: np.ndarray, obs: np.ndarray, nbr: list, w: np.ndarray) -> np.ndarray:
    """M3 이웃 특징: 관측된 이웃만 1/d 가중평균한 a[t], a[t]−a[t−1], a[t]−a[t−3], 관측 이웃 수."""
    cur, oc = val[:, nbr, -1], obs[:, nbr, -1]
    out = []
    for lag in (0, 1, 3):
        x = cur if lag == 0 else cur - val[:, nbr, -1 - lag]
        ok = oc if lag == 0 else oc * obs[:, nbr, -1 - lag]
        den = (ok * w).sum(1)
        out.append(np.where(den > 0, (np.where(ok > 0, x, 0.0) * w).sum(1) / np.maximum(den, 1e-12), 0.0))
    return np.column_stack([*out, oc.sum(1)])


def road_graph(dist: np.ndarray) -> np.ndarray:
    n = len(dist)
    return knn_weights(dist, list(range(n)), list(range(n)), K)


def corr_graph(a: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """학습 구간 10분 변화량 상관으로 받는 센서마다 양수 상관 상위 K개(가중치 = 상관). 두 블록(t−1, t) 모두 학습 구간인 변화만."""
    d = a[1:] - a[:-1]
    rows = train_mask[1:] & train_mask[:-1]
    d = d[rows]
    n = a.shape[1]
    C = np.full((n, n), np.nan)
    ok = np.isfinite(d)
    for i in range(n):
        for j in range(i + 1, n):
            m = ok[:, i] & ok[:, j]
            if m.sum() >= CORR_MIN_PAIRS and d[m, i].std() > 0 and d[m, j].std() > 0:
                C[i, j] = C[j, i] = np.corrcoef(d[m, i], d[m, j])[0, 1]
    W = np.zeros((n, n))
    for r in range(n):
        cand = sorted(((C[r, c], c) for c in range(n) if c != r and np.isfinite(C[r, c]) and C[r, c] > 0), reverse=True)[:K]
        for v, c in cand:
            W[r, c] = v
    return W


# ----------------------------------------------------------------------------- 모델
class STGNN(nn.Module):
    def __init__(self, stat_dim: int, hidden: int = HIDDEN, n_layers: int = 2, n_out: int = len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.enc = nn.Linear(hidden + stat_dim, hidden)
        self.self_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.nbr_lin = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers))
        self.out = nn.Linear(hidden, n_out)

    def forward(self, seq: torch.Tensor, stat: torch.Tensor, m: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """seq (B,N,W,2), stat (B,N,S), m (B,N) t 블록 관측, A (N,N) 행=받는 센서. 반환 (B,N,H) 정규화 변화."""
        B, N, W, _ = seq.shape
        _, hT = self.gru(seq.reshape(B * N, W, 2))
        h = torch.relu(self.enc(torch.cat([hT[-1].reshape(B, N, -1), stat], dim=-1)))
        w = A[None] * m[:, None, :]
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        for sl, nl in zip(self.self_lin, self.nbr_lin):
            h = torch.relu(sl(h) + nl(w @ h))
        return self.out(h)


def targets(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """기점 t 의 목표 a[t+h] − a[t] (B, N, H) 와 정의 여부. t+h 가 자료 밖이면 미정의."""
    B, N = a.shape
    y = np.full((B, N, len(HORIZONS)), np.nan)
    for c, h in enumerate(HORIZONS):
        y[: B - h, :, c] = a[h:] - a[: B - h]
    return y, np.isfinite(y)


def train_gnn(seq, stat, m, y, lab_ok, w_t, A, train_t, seed, epochs: int = EPOCHS):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = STGNN(stat.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    At = torch.tensor(A, dtype=torch.float32)
    for _ in range(epochs):
        perm = train_t[torch.randperm(len(train_t), generator=gen).numpy()]
        for s in range(0, len(perm), BATCH):
            b = perm[s: s + BATCH]
            pred = model(torch.tensor(seq[b]), torch.tensor(stat[b]), torch.tensor(m[b]), At)
            ok = torch.tensor(lab_ok[b], dtype=torch.float32) * torch.tensor(w_t[b], dtype=torch.float32)[:, None, None]
            if float(ok.sum()) <= 0:
                continue
            hub = nn.functional.huber_loss(pred, torch.tensor(np.nan_to_num(y[b]), dtype=torch.float32), reduction="none",
                                           delta=HUBER_DELTA)
            loss = (hub * ok).sum() / ok.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


# ----------------------------------------------------------------------------- 한 폴드
def run_fold(inp: dict, tg: list, eval_idx, train_mask, graphs=("road", "corr"), seeds=SEEDS, ridge=True, epochs: int = EPOCHS):
    """반환: {모델: (len(eval_idx), len(tg), H) 예측 정규화 변화}, scale (N,), a (B,N). 기점 = eval_idx."""
    blk, B = inp["blk"], inp["blk"].shape[0]
    N = blk.shape[1]
    R, scale = normalized_rise(blk, inp["how"], train_mask)
    a = R                                                                  # 정규화 편차
    val, obs = history(a)
    y, yok = targets(a)
    cur_ok = obs[:, :, -1] > 0
    # 학습 기점: 기점과 모든 목표 시각이 학습 구간
    train_t = np.array([t for t in range(WINDOW - 1, B - max(HORIZONS)) if all(train_mask[t + h] for h in (0, *HORIZONS))])
    out = {}
    if ridge:
        dist = inp["dist"]
        for name in ("ar", "ar_rain", "ar_rain_nbr"):
            out[name] = np.full((len(eval_idx), len(tg), len(HORIZONS)), np.nan)
        for c, j in enumerate(tg):
            f_self = self_features(val, obs, j)
            f_rt = np.hstack([inp["rain_feat"][j], inp["time_feat"]])
            nb = [x for _, x in sorted((dist[j, q], q) for q in range(N) if q != j)[:K]]
            f_nb = neighbor_features(val, obs, nb, 1.0 / dist[j, nb])
            feats = {"ar": f_self, "ar_rain": np.hstack([f_self, f_rt]), "ar_rain_nbr": np.hstack([f_self, f_rt, f_nb])}
            for hc in range(len(HORIZONS)):
                rows = train_t[cur_ok[train_t, j] & yok[train_t, j, hc]]
                for name, X in feats.items():
                    fn = _ridge(X[rows], y[rows, j, hc], RIDGE_LAMBDA)
                    out[name][:, c, hc] = fn(X[eval_idx])
    if graphs:
        st = np.concatenate([np.stack(inp["rain_feat"], axis=1), np.repeat(inp["time_feat"][:, None, :], N, axis=1)], axis=2)
        xs = st[train_t].reshape(-1, st.shape[-1])
        mu, sd = xs.mean(0), np.where(xs.std(0) > SD_FLOOR, xs.std(0), 1.0)
        stat = ((st - mu) / sd).astype(np.float32)
        seq = np.stack([val, obs], axis=-1).astype(np.float32)
        m = cur_ok.astype(np.float32)
        lab = np.zeros(N, bool)
        lab[tg] = True
        lab_ok = yok & cur_ok[:, :, None] & lab[None, :, None]
        for c, h in enumerate(HORIZONS):                                  # 목표 시각이 학습 구간 밖이면 손실 제외
            tm = np.zeros(B, bool)
            tm[: B - h] = train_mask[h:]
            lab_ok[:, :, c] &= tm[:, None]
        w_t = event_weight_vector(inp["bt"], inp["events"], EVENT_WEIGHT)
        for g in graphs:
            A = road_graph(inp["dist"]) if g == "road" else corr_graph(a, train_mask)
            preds = []
            for sd_ in seeds:
                model = train_gnn(seq, stat, m, y, lab_ok, w_t, A, train_t, sd_, epochs)
                with torch.no_grad():
                    p = model(torch.tensor(seq[eval_idx]), torch.tensor(stat[eval_idx]), torch.tensor(m[eval_idx]),
                              torch.tensor(A, dtype=torch.float32)).numpy()
                preds.append(p[:, tg, :])
            out[f"gnn_{g}"] = np.mean(preds, axis=0)
    return out, scale, a


def to_levels(pred_norm: dict, scale, eval_idx, tg, blk, store: dict):
    """기점 예측(정규화 변화) → 목표 시각 τ 의 수위(m) 예측. store[name][hc] (B, len(tg))."""
    B = blk.shape[0]
    for name, p in pred_norm.items():
        for hc, h in enumerate(HORIZONS):
            ok = eval_idx + h < B
            t, tau = eval_idx[ok], eval_idx[ok] + h
            store[name][hc][tau] = blk[t][:, tg] + p[ok, :, hc] * scale[tg]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda s: print(s, flush=True)
    roles = load_roles()
    nodes = roles["pool"]
    inp = build_inputs(nodes)
    tg = [nodes.index(n) for n in roles["targets"]]
    B = inp["blk"].shape[0]
    log(f"노드 {len(nodes)}개, 예측 대상 {len(tg)}개, 블록 {B:,}개, 평가 이벤트 {len(inp['events'])}개")
    t0 = time.time()
    if args.timing:
        eval_idx, _, train_mask = next(fold_slices(B))
        run_fold(inp, tg, eval_idx, train_mask, graphs=("road",), seeds=(0,), ridge=False)
        dt = time.time() - t0
        log(f"GNN 1회 {dt:.1f}초 → 전체 {N_FOLDS * len(SEEDS) * 2}회 예상 {dt * N_FOLDS * len(SEEDS) * 2 / 60:.0f}분")
        return
    names = ("ar", "ar_rain", "ar_rain_nbr", "gnn_road", "gnn_corr")
    store = {n: [np.full((B, len(tg)), np.nan) for _ in HORIZONS] for n in ("persist", *names)}
    for f, (eval_idx, _, train_mask) in enumerate(fold_slices(B)):
        tf = time.time()
        pn, scale, _ = run_fold(inp, tg, eval_idx, train_mask)
        pn["persist"] = np.zeros((len(eval_idx), len(tg), len(HORIZONS)))
        to_levels(pn, scale, eval_idx, tg, inp["blk"], store)
        log(f"  fold {f + 1}/{N_FOLDS} ({time.time() - tf:.0f}초)")
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array([nodes[j] for j in tg]),
                        horizons=np.array(HORIZONS), actual_m=inp["blk"][:, tg].astype(np.float32),
                        **{f"pred_{n}_h{h}": store[n][hc].astype(np.float32) for n in store for hc, h in enumerate(HORIZONS)})
    log(f"저장: {OUT_PRED.name} ({time.time() - t0:.0f}초). 판정은 python -m src.analysis.t1_st_eval")


if __name__ == "__main__":
    main()
