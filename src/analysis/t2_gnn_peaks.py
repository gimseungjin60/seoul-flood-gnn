"""첨두 지표 표 (사후 서술 지표, 오차 범위 미산출, 판정에 쓰지 않음): IDW, (4) 강우+이웃(선형), (5) 강우+이웃(트리), GNN 의 이벤트별 첨두 오차(mm)와 첨두 시각 오차(분), 센서 평균.

실행: python -m src.analysis.t2_gnn_peaks   (먼저 t2_loso, t2_gnn 이 끝나 있어야 한다)
출력: data/processed/gangnam_t2_gnn_peak_table.csv, gangnam_t2_gnn_peak_report.txt

정의(기존 t2_loso.summarize 와 같은 유효 블록): 이벤트 창(시작~끝+3시간) 안에서 정답이 있고 이웃 정보가 있는 블록이 6개 이상인 센서만.
첨두 오차 = |예측 최대 − 실제 최대| (mm), 첨두 시각 오차 = |예측 최대 시각 − 실제 최대 시각| (분, 10분 블록). 이벤트별로 센서 평균을 낸다.
test 구간은 읽지 않는다. 이 표는 서술용이며 사전 고정 판정 기준(t2_gnn.py)과 무관하다.
"""
import numpy as np
import pandas as pd

from src.analysis.t2_gnn import OUT_PRED as GNN_PRED
from src.analysis.t2_loso import OUT_PRED as LOSO_PRED, event_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_EVENTS

MODELS = ["idw_p1", "ridge_rain_time_nbr", "hgb_rain_time_nbr", "gnn"]
SHORT = {"idw_p1": "(2) IDW", "ridge_rain_time_nbr": "(4) 선형", "hgb_rain_time_nbr": "(5) 트리", "gnn": "GNN"}
OUT_CSV = PROCESSED / "gangnam_t2_gnn_peak_table.csv"
OUT_TXT = PROCESSED / "gangnam_t2_gnn_peak_report.txt"
MIN_BLOCKS = 6


def peak_table(pred: dict, actual: np.ndarray, bt: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    ok = np.isfinite(actual) & np.isfinite(pred["mean_others"])
    rows = []
    for eid, idx in event_blocks(bt, events):
        for m in MODELS:
            pk, tk = [], []
            for j in range(actual.shape[1]):
                ii = idx[ok[idx, j]]
                if len(ii) < MIN_BLOCKS:
                    continue
                p, a = pred[m][ii, j], actual[ii, j]
                pk.append(abs(np.nanmax(p) - np.nanmax(a)) * 1000)
                tk.append(abs(int(np.nanargmax(p)) - int(np.nanargmax(a))) * 10)
            rows.append({"event_id": eid, "model": m, "n_sensors": len(pk), "peak_err_mm": float(np.mean(pk)), "peak_time_err_min": float(np.mean(tk))})
    return pd.DataFrame(rows)


def main() -> None:
    z = np.load(LOSO_PRED, allow_pickle=True)
    g = np.load(GNN_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    actual = z["actual_m"].astype(float)
    pred = {"mean_others": z["pred_mean_others"].astype(float), "idw_p1": z["pred_idw_p1"].astype(float),
            "ridge_rain_time_nbr": z["pred_ridge_rain_time_nbr"].astype(float), "hgb_rain_time_nbr": z["pred_hgb_rain_time_nbr"].astype(float),
            "gnn": g["pred_gnn"].astype(float)}
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    t = peak_table(pred, actual, bt, ev)
    t.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.3f")
    split = dict(zip(ev.event_id, ev.split_by_start))
    tot = dict(zip(ev.event_id, ev.total_mm))
    L = ["[강남] 첨두 지표 (서술용, 판정에 쓰지 않음). 센서 평균, 이벤트별. test 구간 미사용.",
         "첨두 오차 = |예측 최대 − 실제 최대| (mm) / 첨두 시각 오차 (분)", ""]
    head = f"{'이벤트':<14}" + "".join(f"{SHORT[m]:>16}" for m in MODELS)
    L += [head, "-" * len(head)]
    for eid in sorted(t.event_id.unique()):
        cells = []
        for m in MODELS:
            r = t[(t.event_id == eid) & (t.model == m)].iloc[0]
            cells.append(f"{r.peak_err_mm:7.1f} / {r.peak_time_err_min:5.0f}")
        L.append(f"{eid}번 {tot[eid]:>4.0f}mm {split[eid]:<5}" + "".join(f"{c:>16}" for c in cells))
    cells = []
    for m in MODELS:
        s = t[t.model == m]
        cells.append(f"{s.peak_err_mm.mean():7.1f} / {s.peak_time_err_min.mean():5.0f}")
    L += ["-" * len(head), f"{'이벤트 평균':<14}" + "".join(f"{c:>16}" for c in cells), "",
          "큰 사건(5번 155 mm, 8번 115 mm)의 첨두 오차가 평균을 좌우한다. 이벤트 8개 평균은 서술용이며 CI를 계산하지 않았다."]
    OUT_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
