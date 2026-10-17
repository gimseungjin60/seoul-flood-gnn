"""T2 LOSO 진단: 센서 하나를 통째로 빼고, 이웃 센서의 '현재' 값과 타깃 지점 강우로 그 센서의 기저 대비 상승분을 추정할 수 있는가.
GNN 학습이 아니다(GNN 준비 단계도 아님). 이웃 정보가 강우+시간대 대비 추가 정보를 주는지만 본다.

실행: python -m src.analysis.t2_loso
출력: data/processed/gangnam_t2_loso_by_sensor.csv, gangnam_t2_loso_report.txt, outputs/figures/fig8_t2_loso_gain.png

사전 고정 규칙 (결과를 보기 전에 정함. 결과를 본 뒤 바꾸지 않는다)
[누수 금지] 타깃 센서 i 의 시계열은 어떤 모델의 입력·학습 라벨·특징 계산에도 쓰지 않는다. 학습은 나머지 17개 센서를 각각 '가짜 타깃'으로
  삼고(이웃 후보에서 i 와 자기 자신 제외) 풀링해서 한다. 타깃 센서의 좌표(거리, 최근접 강우량계)는 관측이 아니므로 사용한다.
  예외(라벨 정의에만 사용): '기저 대비 상승분'을 만들려면 타깃의 기저선(훈련 구간 요일x시 평균)과 규모(표준편차)가 필요하다. 이는 정답을
  정의하는 데만 쓰고 모델 입력에는 들어가지 않는다. 즉 '이력이 있던 센서를 가린 경우'의 진단이며, 이력이 전혀 없는 지점은 기저선 자체를 모른다.
[데이터] 10분 평균 블록(관측 5개 이상), train+val 구간(분 인덱스 < 112,608)만. test 는 읽지 않는다.
[검증] 시간순 연속 5개 폴드 교차검증. 평가 폴드 양쪽 24시간(144블록)을 학습에서 제외(purge). 기저선·규모는 폴드마다 학습 구간으로 계산.
[타깃] 정규화 상승분 R = (수준 - 요일x시 평균 프로파일) / max(훈련 표준편차, 0.01 m). 지표는 미터로 환산해서 계산한다.
[지표] 이벤트 구간 MAE(유의 강우 이벤트 시작~끝+3시간 안의 블록; train+val 에 있는 8개 이벤트), 이벤트별 첨두오차(|예측 최대 - 실제 최대|, m),
       첨두시각 오차(분). 보조로 전체 블록 MAE.
[모델] (1) 전체 평균: 나머지 17개 센서 R 의 평균  (2) IDW(p=1, 도로거리)
       (3) 타깃 지점 강우 + 시간대 특징만 Ridge  (4) (3) + 이웃 센서 현재 상승분 Ridge  (5) (4)와 같은 입력의 HistGradientBoosting
       (3b) 대조군(판정에 쓰지 않음): (3)과 같은 입력의 HistGradientBoosting — (5)-(3) 차이가 '이웃 정보'인지 '비선형'인지 가르기 위함.
       Ridge λ=10(표준화된 특징), HGB: max_iter=100, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, l2=1.0,
       early_stopping 없음, random_state=0, 풀링한 학습 행을 하나 걸러 사용(자기상관 완화·속도).
       강우 특징: 같은 시각 구간 포함 최근 6슬롯 log1p(mm), 결측 슬롯 수, 선행 누적 3h/6h/24h log1p(mm)(타깃 지점 최근접 강우량계).
       시간대 특징: 시각 sin/cos, 주말 여부. 이웃 특징: 도로거리 최근접 3개의 상승분과 log 거리, IDW 평균, 전체 평균, 유효 이웃 비율, 최근접 3개 중 결측 수.
[판정] 센서별 개선율 = 1 - 이벤트MAE(모델)/이벤트MAE(3). (4) 또는 (5)가 10% 이상 줄이는 센서가 18개 중 6개 이상이면 'T2 공간 신호 있음'.
       보조로 (4)만, (5)만, 둘 다 충족하는 센서 수와 (5) vs (3b) 도 보고한다.
한계: 이벤트 표본이 8개 사건뿐이고 폴드 교차검증이라 훈련에 비슷한 크기 이벤트가 없는 폴드가 있다. 가짜 타깃 학습 풀이 17개 센서 뿐이다.
"""
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from src.analysis.spatial_signal import PAIR_CSV  # noqa: E402
from src.analysis.spatial_signal_profile_adjusted import hour_of_week, week_profile  # noqa: E402
from src.data.assign_rain_gauges import OUT_CSV as ASSIGN_CSV, node_blocks  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ, VAL_END  # noqa: E402
from src.data.prepare_station_timeseries import OUT_NPZ as SEWER_NPZ  # noqa: E402

OUT_SENSOR = PROCESSED / "gangnam_t2_loso_by_sensor.csv"
OUT_PRED = PROCESSED / "gangnam_t2_loso_predictions.npz"   # 재생성 가능(약 1분), git 제외
REPORT = PROCESSED / "gangnam_t2_loso_report.txt"
N_FOLDS = 5
PURGE = 144                      # 24시간(10분 블록)
SCALE_FLOOR_M = 0.01
RIDGE_LAMBDA = 10.0
HGB_STRIDE = 2
EVENT_AFTER_H = 3
IMPROVE = 0.10
MIN_SENSORS = 6
N_RECENT, CUM_SLOTS = 6, (18, 36, 144)
MODELS = ["mean_others", "idw_p1", "ridge_rain_time", "ridge_rain_time_nbr", "hgb_rain_time_nbr", "hgb_rain_time"]
LABEL = {"mean_others": "(1) 전체 평균", "idw_p1": "(2) IDW", "ridge_rain_time": "(3) 강우+시간대 Ridge",
         "ridge_rain_time_nbr": "(4) (3)+이웃 Ridge", "hgb_rain_time_nbr": "(5) (4)입력 HGB", "hgb_rain_time": "(3b) 강우+시간대 HGB[대조]"}


@dataclass
class Params:
    ridge_lambda: float = RIDGE_LAMBDA
    hgb_iter: int = 100
    hgb_stride: int = HGB_STRIDE


# ----------------------------------------------------------------------------- 특징
def nowcast_rain_features(rain: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """rain (S,) mm/10분 전체 격자(NaN=결측), pos (B,) 각 블록과 같은 구간의 슬롯 인덱스.
    같은 구간 포함 최근 6슬롯 log1p, 그중 결측 수, 선행 누적(3h,6h,24h, 같은 슬롯 포함) log1p. 반환 (B, 10)."""
    if (pos < max(CUM_SLOTS) - 1).any():
        raise ValueError("선행 24시간 누적에 필요한 이전 슬롯이 부족하다")
    r0 = np.nan_to_num(rain, nan=0.0)
    miss = np.isnan(rain).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(r0)])
    recent = np.stack([r0[pos - j] for j in range(N_RECENT)], axis=1)
    n_miss = sum(miss[pos - j] for j in range(N_RECENT))[:, None]
    cum = np.stack([cs[pos + 1] - cs[pos + 1 - n] for n in CUM_SLOTS], axis=1)
    return np.hstack([np.log1p(recent), n_miss, np.log1p(cum)])


def time_features(starts: pd.DatetimeIndex) -> np.ndarray:
    h = (starts.hour + starts.minute / 60).to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), (starts.dayofweek >= 5).astype(float)], axis=1)


def neighbor_features(R_c: np.ndarray, d: np.ndarray):
    """R_c (B, M) 후보 이웃의 현재 정규화 상승분(NaN=결측), d (M,) 타깃(또는 가짜 타깃)과의 도로거리 m.
    반환: 특징 (B, 10), 유효 이웃 수 (B,). 유효 이웃이 없는 행은 호출 쪽에서 버린다."""
    ok = np.isfinite(R_c)
    n_valid = ok.sum(axis=1)
    w = 1.0 / d
    num = np.where(ok, R_c, 0.0) @ w
    den = ok @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        idw = np.where(den > 0, num / den, np.nan)
        mean_c = np.where(n_valid > 0, np.where(ok, R_c, 0.0).sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    near = np.argsort(d)[:3]
    fill = np.nan_to_num(idw, nan=0.0)
    near_r = np.where(ok[:, near], R_c[:, near], fill[:, None])
    n_miss = (~ok[:, near]).sum(axis=1)
    logd = np.tile(np.log(d[near] / 1000.0), (R_c.shape[0], 1))
    feats = np.hstack([near_r, logd, fill[:, None], np.nan_to_num(mean_c, nan=0.0)[:, None],
                       (n_valid / R_c.shape[1])[:, None], n_miss[:, None]])
    return feats, n_valid


# ----------------------------------------------------------------------------- 학습·예측 (타깃 열은 여기서 즉시 제거된다)
def _ridge(Xtr, ytr, lam):
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Z = (Xtr - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
    return lambda X: ((X - mu) / sd) @ w + ym


def _hgb(Xtr, ytr, n_iter):
    m = HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100,
                                      l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict


def loso_predict(R: np.ndarray, dist: np.ndarray, rain_feat: list, time_feat: np.ndarray, i: int,
                 train_idx: np.ndarray, eval_idx: np.ndarray, params: Params | None = None) -> dict:
    """타깃 센서 i 를 뺀 LOSO 예측. 반환: {모델: (len(eval_idx),) 정규화 상승분 예측, NaN=이웃 정보 없음}.
    R (B, N) 에서 i 열은 가장 먼저 제거되어 이후 어떤 계산에도 쓰이지 않는다(누수 방지 테스트가 이를 검증)."""
    params = params or Params()
    n = R.shape[1]
    others = [j for j in range(n) if j != i]
    R_oth = np.delete(R, i, axis=1)                                   # (B, N-1): 타깃 열 제거
    pos_in = {j: c for c, j in enumerate(others)}

    def pooled_rows():
        """가짜 타깃 k(≠i)별로 학습 행을 쌓는다. 이웃 후보 = others - {k}."""
        Xa, Xb, y = [], [], []
        for k in others:
            cand = [j for j in others if j != k]
            Rk = R_oth[:, [pos_in[j] for j in cand]]
            nb, nv = neighbor_features(Rk[train_idx], dist[k, cand])
            label = R_oth[train_idx, pos_in[k]]
            keep = np.isfinite(label) & (nv > 0)
            base = np.hstack([rain_feat[k][train_idx], time_feat[train_idx]])
            Xa.append(base[keep])
            Xb.append(np.hstack([base, nb])[keep])
            y.append(label[keep])
        return np.vstack(Xa), np.vstack(Xb), np.concatenate(y)
    Xa, Xb, y = pooled_rows()
    sub = slice(None, None, params.hgb_stride)
    models = {"ridge_rain_time": _ridge(Xa, y, params.ridge_lambda), "ridge_rain_time_nbr": _ridge(Xb, y, params.ridge_lambda),
              "hgb_rain_time_nbr": _hgb(Xb[sub], y[sub], params.hgb_iter), "hgb_rain_time": _hgb(Xa[sub], y[sub], params.hgb_iter)}

    R_eval = R_oth[eval_idx]
    nb_e, nv_e = neighbor_features(R_eval, dist[i, others])
    base_e = np.hstack([rain_feat[i][eval_idx], time_feat[eval_idx]])
    Xb_e = np.hstack([base_e, nb_e])
    ok = nv_e > 0
    out = {}
    ok_f = np.isfinite(R_eval)
    w = 1.0 / dist[i, others]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mean_others"] = np.where(ok, np.where(ok_f, R_eval, 0.0).sum(axis=1) / np.maximum(ok_f.sum(axis=1), 1), np.nan)
        out["idw_p1"] = np.where(ok, (np.where(ok_f, R_eval, 0.0) @ w) / np.maximum(ok_f @ w, 1e-12), np.nan)
    for name, f in models.items():
        X = base_e if name in ("ridge_rain_time", "hgb_rain_time") else Xb_e
        p = np.full(len(eval_idx), np.nan)
        p[ok] = f(X[ok])
        out[name] = p
    return out


# ----------------------------------------------------------------------------- 준비·평가
def build_inputs():
    s = np.load(SEWER_NPZ, allow_pickle=True)
    nodes = [str(x) for x in s["node_ids"]]
    blk = node_blocks(s["values"][:VAL_END], s["mask"][:VAL_END])
    t10 = pd.DatetimeIndex(s["times"][:VAL_END])[: len(blk) * 10]
    bt = t10[9::10]                                                    # 블록 끝 라벨(:09...) = 강우 격자 라벨
    z = np.load(RAIN_NPZ, allow_pickle=True)
    rt = pd.DatetimeIndex(z["times"])
    gids = [str(g) for g in z["gauge_ids"]]
    pos = rt.get_indexer(bt)
    if (pos < 0).any():
        raise ValueError("수위 블록 라벨이 강우 격자에 없다")
    asg = pd.read_csv(ASSIGN_CSV, dtype={"gauge_id": str}).set_index("node_id").loc[nodes]
    rain_feat = [nowcast_rain_features(z["values"][:, gids.index(asg.loc[nd, "gauge_id"])].astype(float), pos) for nd in nodes]
    nid = {k: c for c, k in enumerate(nodes)}
    dist = np.zeros((len(nodes), len(nodes)))
    for r in pd.read_csv(PAIR_CSV).itertuples():
        dist[nid[r.node_a], nid[r.node_b]] = dist[nid[r.node_b], nid[r.node_a]] = r.road_dist_m
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    return dict(nodes=nodes, blk=blk, bt=bt, rain_feat=rain_feat, time_feat=time_features(bt - pd.Timedelta(minutes=9)),
                how=hour_of_week(bt - pd.Timedelta(minutes=9)), dist=dist, events=ev)


def fold_slices(B: int):
    edges = np.linspace(0, B, N_FOLDS + 1).astype(int)
    for f in range(N_FOLDS):
        a, b = edges[f], edges[f + 1]
        tr = np.ones(B, dtype=bool)
        tr[max(0, a - PURGE): min(B, b + PURGE)] = False
        yield np.arange(a, b), np.flatnonzero(tr), tr


def normalized_rise(blk, how, train_mask):
    """폴드 학습 구간으로 기저선(요일x시 평균)과 규모를 만들어 정규화 상승분과 규모(m)를 반환."""
    prof = week_profile(blk, how, train_mask, "mean")
    anom = blk - prof[how]
    sd = np.nanstd(np.where(train_mask[:, None], anom, np.nan), axis=0)
    scale = np.maximum(sd, SCALE_FLOOR_M)
    return anom / scale, scale


def run_loso(inp: dict, params: Params | None = None, log=None):
    blk, nodes = inp["blk"], inp["nodes"]
    B, N = blk.shape
    pred_m = {m: np.full((B, N), np.nan) for m in MODELS}
    act_m = np.full((B, N), np.nan)
    for f, (eval_idx, train_idx, train_mask) in enumerate(fold_slices(B)):
        R, scale = normalized_rise(blk, inp["how"], train_mask)
        t0 = time.time()
        for i in range(N):
            out = loso_predict(R, inp["dist"], inp["rain_feat"], inp["time_feat"], i, train_idx, eval_idx, params)
            for m in MODELS:
                pred_m[m][eval_idx, i] = out[m] * scale[i]
            act_m[eval_idx, i] = R[eval_idx, i] * scale[i]
        if log:
            log(f"  fold {f + 1}/{N_FOLDS} 완료 ({time.time() - t0:.0f}초)")
    return pred_m, act_m


def event_blocks(bt: pd.DatetimeIndex, ev: pd.DataFrame):
    """이벤트별 블록 인덱스(시작~끝+3h). 평가 구간 밖(test)에 있는 이벤트는 제외."""
    out = []
    for e in ev.itertuples():
        idx = np.flatnonzero((bt >= e.start) & (bt <= e.end + pd.Timedelta(hours=EVENT_AFTER_H)))
        if len(idx) and bt[0] <= e.start and bt[-1] >= e.end:
            out.append((e.event_id, idx))
    return out


def summarize(pred_m, act_m, bt, events, nodes):
    ev_idx = event_blocks(bt, events)
    in_ev = np.zeros(len(bt), dtype=bool)
    for _, idx in ev_idx:
        in_ev[idx] = True
    rows = []
    for j, nd in enumerate(nodes):
        a = act_m[:, j]
        base_ok = np.isfinite(a) & np.isfinite(pred_m["mean_others"][:, j])
        r = {"sensor": nd, "n_event_blocks": int((base_ok & in_ev).sum()), "n_all_blocks": int(base_ok.sum())}
        for m in MODELS:
            p = pred_m[m][:, j]
            e = np.abs(p - a)
            r[f"event_mae_{m}"] = float(e[base_ok & in_ev].mean())
            r[f"all_mae_{m}"] = float(e[base_ok].mean())
            pk, tk = [], []
            for _, idx in ev_idx:
                ii = idx[base_ok[idx]]
                if len(ii) < 6:
                    continue
                pk.append(abs(np.nanmax(p[ii]) - np.nanmax(a[ii])))
                tk.append(abs(int(np.nanargmax(p[ii])) - int(np.nanargmax(a[ii]))) * 10)
            r[f"peak_err_{m}"] = float(np.median(pk)) if pk else np.nan
            r[f"peak_time_err_min_{m}"] = float(np.median(tk)) if tk else np.nan
        for m in ("ridge_rain_time_nbr", "hgb_rain_time_nbr"):
            r[f"gain_{m}_vs_3"] = 1 - r[f"event_mae_{m}"] / r["event_mae_ridge_rain_time"]
        r["gain_hgb_nbr_vs_3b"] = 1 - r["event_mae_hgb_rain_time_nbr"] / r["event_mae_hgb_rain_time"]
        r["pass_4_or_5"] = bool(r["gain_ridge_rain_time_nbr_vs_3"] >= IMPROVE or r["gain_hgb_rain_time_nbr_vs_3"] >= IMPROVE)
        rows.append(r)
    return pd.DataFrame(rows), len(ev_idx)


def verdict(df: pd.DataFrame) -> dict:
    n4 = int((df.gain_ridge_rain_time_nbr_vs_3 >= IMPROVE).sum())
    n5 = int((df.gain_hgb_rain_time_nbr_vs_3 >= IMPROVE).sum())
    n_or = int(df.pass_4_or_5.sum())
    n_and = int(((df.gain_ridge_rain_time_nbr_vs_3 >= IMPROVE) & (df.gain_hgb_rain_time_nbr_vs_3 >= IMPROVE)).sum())
    n5b = int((df.gain_hgb_nbr_vs_3b >= IMPROVE).sum())
    ok = n_or >= MIN_SENSORS
    line = (f"T2 공간 신호 {'있음' if ok else '확인되지 않음'}: (4) 또는 (5)가 (3) 대비 이벤트 MAE를 10% 이상 줄인 센서 {n_or}/{len(df)}개 "
            f"(기준 {MIN_SENSORS}개 이상)")
    return {"n4": n4, "n5": n5, "n_or": n_or, "n_and": n_and, "n5b": n5b, "ok": ok, "line": line}


def make_figure(df: pd.DataFrame, v: dict):
    from src.visualization.make_figures import C, _note, _save
    d = df.assign(best=df[["gain_ridge_rain_time_nbr_vs_3", "gain_hgb_rain_time_nbr_vs_3"]].max(axis=1)).sort_values("best")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(d))
    h = 0.38
    ax.barh(y - h / 2, d.gain_ridge_rain_time_nbr_vs_3 * 100, height=h, color=C["s1"], label="(4) 강우+시간대+이웃 Ridge", zorder=3)
    ax.barh(y + h / 2, d.gain_hgb_rain_time_nbr_vs_3 * 100, height=h, color=C["s2"], label="(5) 같은 입력 HGB", zorder=3)
    ax.axvline(0, color=C["ink2"], linewidth=1.2, zorder=4)
    ax.axvline(IMPROVE * 100, color=C["ink"], linewidth=1.2, linestyle="--", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(d.sensor)
    ax.set_xlabel("이벤트 구간 MAE 개선율 (%), 기준 (3) 강우+시간대만 대비 (양수가 좋음)")
    ax.set_title(f"이웃 센서를 쓰면 센서 없는 지점의 이벤트 오차가 줄어드는가: {v['n_or']}/{len(d)}개 센서가 10% 이상")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    _note(fig, "센서 1개를 통째로 제외(LOSO), 타깃 과거값 미사용. 시간순 5폴드 교차검증, train+val의 이벤트 8개. 점선=사전 기준 10%.\n"
               "주의: 기준 (3)은 약한 기준선이다. 단순 이웃 평균(1)·IDW(2)도 (4)와 같은 수준이며 학습 모델이 이들보다 낫지는 않다(docs/08).", y=-0.045)
    _save(fig, "fig8_t2_loso_gain.png")


def main() -> None:
    t0 = time.time()
    inp = build_inputs()
    log = lambda s: print(s, flush=True)
    log(f"블록 {inp['blk'].shape[0]:,}개, 센서 {inp['blk'].shape[1]}개, 폴드 {N_FOLDS}")
    pred_m, act_m = run_loso(inp, Params(), log)
    np.savez_compressed(OUT_PRED, bt=inp["bt"].to_numpy().astype("datetime64[s]"), nodes=np.array(inp["nodes"]),
                        models=np.array(MODELS), actual_m=act_m.astype(np.float32),
                        **{f"pred_{m}": pred_m[m].astype(np.float32) for m in MODELS})
    df, n_ev = summarize(pred_m, act_m, inp["bt"], inp["events"], inp["nodes"])
    df.to_csv(OUT_SENSOR, index=False, encoding="utf-8-sig", float_format="%.5f")
    v = verdict(df)
    make_figure(df, v)

    L = []
    w = L.append
    w("[강남] T2 LOSO 진단: 센서 하나를 통째로 뺀 추정 (GNN 아님, test 구간 미사용)")
    w("=" * 78)
    w("※ 사전 고정 규칙은 src/analysis/t2_loso.py 상단 docstring 참조(결과 확인 전 작성). 타깃 센서 시계열은 입력·학습 라벨에 쓰지 않는다(라벨 정의의 기저선·규모 제외).")
    w(f"평가: train+val 시간순 {N_FOLDS}폴드 교차검증, 이벤트 {n_ev}개(블록 {int(df.n_event_blocks.max())}개 내외), 전체 블록 {int(df.n_all_blocks.max()):,}개 내외")
    w("")
    w(f"판정: {v['line']}")
    w(f"  보조: (4)만 {v['n4']}개, (5)만 {v['n5']}개, (4)·(5) 둘 다 {v['n_and']}개, (5)가 (3b: 같은 입력에서 이웃만 뺀 HGB) 대비 10% 이상 {v['n5b']}개")
    w("")
    w("1. 이벤트 구간 MAE (m) 센서별, 모델별")
    cols = [f"event_mae_{m}" for m in MODELS]
    t = df.set_index("sensor")[cols].rename(columns={c: LABEL[c.replace("event_mae_", "")][:5] for c in cols})
    w(t.to_string(float_format=lambda x: f"{x:.4f}"))
    w(f"  평균: " + ", ".join(f"{LABEL[m][:5]} {df[f'event_mae_{m}'].mean():.4f}" for m in MODELS))
    w("")
    w("2. (3) 대비 개선율 (이벤트 MAE, 양수가 좋음) 과 판정")
    g = df.set_index("sensor")[["gain_ridge_rain_time_nbr_vs_3", "gain_hgb_rain_time_nbr_vs_3", "gain_hgb_nbr_vs_3b", "pass_4_or_5"]]
    g.columns = ["(4)vs(3)", "(5)vs(3)", "(5)vs(3b)", "기준충족"]
    w(g.to_string(float_format=lambda x: f"{x:+.3f}"))
    w(f"  중앙값: (4) {df.gain_ridge_rain_time_nbr_vs_3.median():+.3f}, (5) {df.gain_hgb_rain_time_nbr_vs_3.median():+.3f}, (5) vs (3b) {df.gain_hgb_nbr_vs_3b.median():+.3f}")
    w("")
    w("3. 첨두오차(m, 이벤트별 |예측 최대-실제 최대| 의 센서 내 중앙값)와 첨두시각 오차(분)")
    for key, title in (("peak_err", "첨두오차(m)"), ("peak_time_err_min", "첨두시각 오차(분)")):
        cc = [f"{key}_{m}" for m in MODELS]
        tt = df.set_index("sensor")[cc].rename(columns={c: LABEL[c.replace(key + '_', '')][:5] for c in cc})
        w(f"  [{title}] 센서 중앙값: " + ", ".join(f"{LABEL[m][:5]} {df[f'{key}_{m}'].median():.3f}" for m in MODELS))
    w("")
    w("4. 전체 블록 MAE (m, 참고)")
    ca = [f"all_mae_{m}" for m in MODELS]
    w("  평균: " + ", ".join(f"{LABEL[m][:5]} {df[f'all_mae_{m}'].mean():.4f}" for m in MODELS))
    w("")
    w("주의")
    w("  - 이력이 있던 센서를 가린 진단이다. 라벨의 기저선·규모는 타깃의 훈련 구간으로 만들었다(입력 아님). 이력 없는 지점은 기저선부터 모른다.")
    w("  - 이벤트는 8개 사건이고 폴드마다 평가 이벤트와 비슷한 크기의 훈련 이벤트가 있다는 보장이 없다. 센서별 개선율은 노이즈가 크다.")
    w("  - 10분 블록은 서로 독립이 아니다. 판정은 센서 수 기준이며 신뢰구간이 아니다.")
    w("  - (5)가 (3)보다 낫더라도 (3b)와 비교해야 '이웃 정보'와 '비선형'을 구분할 수 있다.")
    w(f"처리 시간 {time.time() - t0:.0f}초")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
