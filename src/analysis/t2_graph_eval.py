"""2026 그래프 보완 실험 판정: 사전 고정 규칙(src/analysis/t2_graph_rules.py, docs/14)의 Q1~Q3만 판정하고 나머지는 참고로 보고한다.
결과를 보기 전에 작성했다(2026-11-09).

실행: python -m src.analysis.t2_graph_eval   (먼저 t2_2026 --part loso / gnn, t2_graph --part elev / rand 가 끝나야 한다)
출력: data/processed/gangnam2026_t2_graph_report.txt, gangnam2026_t2_graph_ci.csv
비교 기준(IDW, 도로거리 GNN)은 2026 재비교 예측 파일을 그대로 쓴다. 부트스트랩·유효 블록 정의도 t2_2026_eval 과 같다.
"""
import numpy as np
import pandas as pd

from src.analysis.t2_2026 import OUT_PRED as PRED_2026
from src.analysis.t2_2026_eval import load_preds
from src.analysis.t2_2026_rules import BOOT_SEED, N_BOOT
from src.analysis.t2_gnn_eval import NAMES, error_sums
from src.analysis.t2_graph import OUT_DIAG, OUT_PRED
from src.analysis.t2_graph_rules import q1_verdict, q2_verdict, q3_verdict
from src.analysis.t2_loso_bootstrap import bootstrap, ci, macro_mae
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

OUT_REPORT = PROCESSED / "gangnam2026_t2_graph_report.txt"
OUT_CI = PROCESSED / "gangnam2026_t2_graph_ci.csv"
COLS = ["idw_p1", "gnn_road", "gnn_elev", "gnn_rand"]
LABEL = {"idw_p1": "IDW", "gnn_road": "도로거리 GNN", "gnn_elev": "G1 고도 방향 GNN", "gnn_rand": "G2 랜덤 GNN"}


def load_all():
    bt, nodes, actual, pred = load_preds(PRED_2026["loso"], PRED_2026["gnn"])
    for g in ("elev", "rand"):
        z = np.load(OUT_PRED[g], allow_pickle=True)
        if not (np.array_equal(z["bt"], bt.to_numpy().astype("datetime64[s]")) and [str(n) for n in z["nodes"]] == nodes):
            raise ValueError(f"{g}: 시각·센서 순서가 2026 재비교 예측과 다르다")
        if not np.allclose(np.nan_to_num(actual), np.nan_to_num(z["actual_m"].astype(float)), atol=1e-4):
            raise ValueError(f"{g}: 정답이 2026 재비교와 다르다")
        pred[f"gnn_{g}"] = z["pred_gnn"].astype(float)
    pred["gnn_road"] = pred["gnn"]
    return bt, nodes, actual, pred


def stacked_sums(pred, actual, bt, events):
    """COLS 순서의 (E, J, M) 오차 합. error_sums 는 NAMES 기준이라 'gnn' 자리에 그래프별 예측을 넣어 한 번씩 부른다."""
    N0, ev_ids, cols = None, None, {}
    for c in COLS:
        p = dict(pred)
        p["gnn"] = pred[c] if c.startswith("gnn") else pred["gnn"]
        S, N, ev_ids = error_sums(p, actual, bt, events)
        if N0 is not None and not np.array_equal(N, N0):
            raise ValueError("그래프마다 유효 블록 수가 다르다")
        N0 = N
        cols[c] = S[:, :, NAMES.index("gnn" if c.startswith("gnn") else c)]
    return np.stack([cols[c] for c in COLS], axis=2), N0, ev_ids


def main() -> None:
    bt, nodes, actual, pred = load_all()
    events = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    events = events[events.is_eval].reset_index(drop=True)
    S, N, ev_ids = stacked_sums(pred, actual, bt, events)
    E, J = S.shape[0], S.shape[1]
    k = {c: i for i, c in enumerate(COLS)}
    point, mae_j = macro_mae(S, N, np.ones((1, E)))
    point, mae_j = point[0], mae_j[0]
    boot, _ = bootstrap(S, N, N_BOOT, BOOT_SEED)

    def diff(a, b):
        lo, hi = ci(boot[:, k[a]] - boot[:, k[b]])
        return float(point[k[a]] - point[k[b]]), lo, hi

    q1, q2, q3 = diff("gnn_elev", "gnn_road"), diff("gnn_elev", "idw_p1"), diff("gnn_road", "gnn_rand")
    v1, v2, v3 = q1_verdict(*q1[1:]), q2_verdict(*q2[1:]), q3_verdict(*q3[1:])
    ref = diff("gnn_rand", "idw_p1")

    rows = [{"stat": f"mae_{c}", "point_m": point[k[c]], "lo_m": ci(boot[:, k[c]])[0], "hi_m": ci(boot[:, k[c]])[1]} for c in COLS]
    rows += [{"stat": s, "point_m": d[0], "lo_m": d[1], "hi_m": d[2]} for s, d in
             (("Q1_elev_minus_road", q1), ("Q2_elev_minus_idw", q2), ("Q3_road_minus_rand", q3), ("ref_rand_minus_idw", ref))]
    pd.DataFrame(rows).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.5f")

    f = lambda d: f"{d[0] * 1000:+.2f} mm [{d[1] * 1000:+.2f}, {d[2] * 1000:+.2f}]"
    L = ["[강남 2026] 그래프 보완 실험 판정 (사전 고정 규칙: src/analysis/t2_graph_rules.py, docs/14)", "=" * 78,
         f"범위: 서울시 강남구 센서 {J}개(타깃) + 이웃 후보, 2026-06-01~08-31. 평가 이벤트 {E}개 {ev_ids}. 부트스트랩 {N_BOOT}회, 시드 {BOOT_SEED}.",
         "새로 비교한 그래프는 2개(G1 고도 방향, G2 랜덤 대조군)이며 다중 비교 보정은 하지 않았다.",
         "이 검증은 타깃의 기저선·규모를 안다는 가정 하의 상대 상승폭 추정이다. 그래프는 어느 것도 실제 관로 연결이 아니다.",
         "", "[판정]",
         f"  Q1 G1 고도 방향 GNN − 도로거리 GNN: {f(q1)} → {v1}",
         f"  Q2 G1 고도 방향 GNN − IDW: {f(q2)} → {v2}",
         f"  Q3 도로거리 GNN − G2 랜덤 GNN: {f(q3)} → {v3}",
         "", "[참고: 이벤트 MAE (mm), 95% CI]"]
    for c in COLS:
        lo, hi = ci(boot[:, k[c]])
        L.append(f"  {LABEL[c]:<14} {point[k[c]] * 1000:7.2f}  [{lo * 1000:.2f}, {hi * 1000:.2f}]")
    L.append(f"  참고: G2 랜덤 GNN − IDW {f(ref)}")
    L += ["", "[참고: 센서별 이벤트 MAE (mm)]"]
    for j, n in enumerate(nodes):
        L.append(f"  {n}: " + ", ".join(f"{LABEL[c]} {mae_j[j, k[c]] * 1000:.2f}" for c in COLS))
    if OUT_DIAG.exists():
        L += ["", "[참고: 실행 전 진단]", OUT_DIAG.read_text(encoding="utf-8").rstrip()]
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
