"""예측 입력·정답·평가 공용 유틸 (기준 모델과 이후 GNN이 같은 규칙을 쓰도록).

입력 표현: 노드별 2채널
  x_value = (원본 수위 - 학습평균) / 학습표준편차, 결측(NaN)은 0으로 대체(= 학습 평균 수준)
  x_obs   = 원본 관측 마스크(1 관측 / 0 결측) — 대체값과 실제 관측값을 구분
  → 모델 입력에 NaN이 들어가지 않는다. 대체값은 x_obs=0으로 식별된다.
정답·평가: 원본 마스크가 1인 (t+h, 노드)만 사용. 결측 정답은 손실·지표에서 제외(0으로 채워 계산하지 않음).
"""
import numpy as np
import pandas as pd

from src.data.build_forecast_dataset import HORIZONS, OUT_NORM, OUT_SAMPLES, W
from src.data.prepare_station_timeseries import OUT_NPZ


def load_base():
    z = np.load(OUT_NPZ)
    values = z["values"].astype(np.float64)
    mask = z["mask"].astype(bool)
    nodes = list(z["node_ids"])
    norm = pd.read_csv(OUT_NORM).set_index("node_id").loc[nodes]
    return values, mask, nodes, norm["train_mean"].to_numpy(), norm["train_std"].to_numpy()


def load_samples(h: int) -> pd.DataFrame:
    s = pd.read_csv(OUT_SAMPLES[h])
    for c in ("strict_raw", "strict_locf5", "masked_any_target"):
        s[c] = s[c].astype(str).eq("True")
    return s


def verify_samples(s: pd.DataFrame, mask: np.ndarray, h: int, w: int = W) -> dict:
    """샘플 파일의 플래그를 원본 마스크에서 독립 재계산(rolling)해 대조."""
    m = pd.DataFrame(mask.astype(int))
    full = (m.rolling(w).sum() == w).all(axis=1).to_numpy()
    t = s["t_idx"].to_numpy()
    tgt = mask[t + h]
    out = {
        "n_target_observed": bool((tgt.sum(axis=1) == s["n_target_observed"]).all()),
        "strict_raw": bool(((full[t] & tgt.all(axis=1)) == s["strict_raw"]).all()),
        "masked_any_target": bool((tgt.any(axis=1) == s["masked_any_target"]).all()),
        "split_no_overlap": bool(_splits_disjoint(s, h, w)),
        "count_le_total": bool((s.groupby("split")[["strict_raw", "strict_locf5", "masked_any_target"]].sum()
                                .le(s.groupby("split").size(), axis=0)).all().all()),
    }
    return out


def _splits_disjoint(s: pd.DataFrame, h: int, w: int) -> bool:
    """분할별 [입력 시작, 정답] 시간 범위가 서로 겹치지 않는지."""
    r = []
    for name in ("train", "val", "test"):
        x = s[s["split"] == name]
        r.append((x["t_idx"].min() - w + 1, x["t_idx"].max() + h))
    return all(r[i][1] < r[i + 1][0] for i in range(len(r) - 1))


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.where(std > 0, std, 1.0)


def make_inputs(values: np.ndarray, mask: np.ndarray, mean, std, t_idx: np.ndarray, w: int = W):
    """anchor t들의 입력 텐서. 반환 x_value, x_obs: (샘플, w, 노드). NaN 없음."""
    zn = np.where(mask, normalize(values, mean, std), 0.0)
    idx = t_idx[:, None] + np.arange(-w + 1, 1)[None, :]
    return zn[idx], mask[idx].astype(np.float32)


def make_targets(values: np.ndarray, mask: np.ndarray, t_idx: np.ndarray, h: int):
    """정답(m 단위)과 정답 유효 마스크. 결측 정답은 값 0 + 마스크 0 (값은 손실에서 쓰이지 않음)."""
    y_mask = mask[t_idx + h]
    return np.where(y_mask, values[t_idx + h], 0.0), y_mask


def masked_mse(pred: np.ndarray, y: np.ndarray, y_mask: np.ndarray) -> float:
    """유효 정답만 평균. 유효 정답이 하나도 없으면 0.0(손실 기여 없음). NaN 전파 없음."""
    m = y_mask.astype(bool)
    n = m.sum()
    if n == 0:
        return 0.0
    d = np.where(m, np.nan_to_num(pred) - np.nan_to_num(y), 0.0)
    return float((d ** 2).sum() / n)


def metrics(pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> dict:
    """m=True인 칸만 MAE/RMSE(m)."""
    e = (pred - y)[m]
    return {"n_pairs": int(m.sum()), "MAE": float(np.abs(e).mean()) if e.size else np.nan,
            "RMSE": float(np.sqrt((e ** 2).mean())) if e.size else np.nan}


__all__ = ["HORIZONS", "W", "load_base", "load_samples", "verify_samples", "make_inputs", "make_targets",
           "masked_mse", "metrics", "normalize"]
