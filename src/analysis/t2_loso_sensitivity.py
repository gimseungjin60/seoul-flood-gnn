"""T2 LOSO 민감도 (사후 분석, 판정 변경 없음): 23-0014 제외, 큰 사건(5, 8) 제외, 둘 다 제외.

실행: python -m src.analysis.t2_loso_sensitivity   (t2_loso 예측 npz 필요)
출력: data/processed/gangnam_t2_loso_sensitivity_report.txt

각 시나리오에서 모델별 이벤트 MAE(센서 평균) 점추정과 95% CI, IDW 대비 (4)-(2), (5)-(2) 차이 CI(이벤트 복원추출 2000회, 시드 0),
모델 순위를 낸다. 사전 판정(10/18 통과)은 바꾸지 않는다. 서울시 강남구 센서 18개(또는 제외 후 17개) 기준이며 타깃 기저선·규모를 안다는
가정 하의 상대 상승폭 추정이다.
"""
import numpy as np
import pandas as pd

from src.analysis.t2_loso import MODELS, OUT_PRED
from src.analysis.t2_loso_bootstrap import SHORT, bootstrap, ci, event_error_sums, macro_mae, pass_count
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall import OUT_EVENTS

REPORT = PROCESSED / "gangnam_t2_loso_sensitivity_report.txt"
DROP_SENSOR = "23-0014"
DROP_EVENTS = (5, 8)
SHOW = ["mean_others", "idw_p1", "ridge_rain_time", "ridge_rain_time_nbr", "hgb_rain_time_nbr", "hgb_rain_time"]


def subset(S, N, ev_ids, nodes, drop_sensor=None, drop_events=()):
    ev_keep = [i for i, e in enumerate(ev_ids) if e not in drop_events]
    sen_keep = [j for j, n in enumerate(nodes) if n != drop_sensor]
    return S[np.ix_(ev_keep, sen_keep)], N[np.ix_(ev_keep, sen_keep)], [ev_ids[i] for i in ev_keep], [nodes[j] for j in sen_keep]


def run_scenario(S, N):
    k = {m: i for i, m in enumerate(MODELS)}
    point_macro, point_sensor = macro_mae(S, N, np.ones((1, S.shape[0])))
    point = point_macro[0]
    boot_macro, boot_sensor = bootstrap(S, N)
    cis = {m: ci(boot_macro[:, k[m]]) for m in MODELS}
    d4 = ci(boot_macro[:, k["ridge_rain_time_nbr"]] - boot_macro[:, k["idw_p1"]])
    d5 = ci(boot_macro[:, k["hgb_rain_time_nbr"]] - boot_macro[:, k["idw_p1"]])
    rank = sorted(SHOW[:5], key=lambda m: point[k[m]])
    return dict(point=point, cis=cis, d4=d4, d5=d5, rank=rank, pass_point=int(pass_count(point_sensor)[0]),
                pass_ci=(int(np.percentile(pass_count(boot_sensor), 2.5)), int(np.percentile(pass_count(boot_sensor), 97.5))),
                k=k, d4_point=point[k["ridge_rain_time_nbr"]] - point[k["idw_p1"]], d5_point=point[k["hgb_rain_time_nbr"]] - point[k["idw_p1"]])


def verdict_text(lo, hi):
    return "구분 불가(CI가 0 포함)" if lo <= 0 <= hi else ("IDW가 낮음" if lo > 0 else "(4)/(5)가 낮음")


def main() -> None:
    z = np.load(OUT_PRED, allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    pred = {m: z[f"pred_{m}"].astype(float) for m in MODELS}
    nodes = [str(x) for x in z["nodes"]]
    S, N, ev_ids = event_error_sums(pred, z["actual_m"].astype(float), bt, pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"]))
    scen = {
        "기준(전체 18개 센서, 이벤트 8개)": subset(S, N, ev_ids, nodes),
        f"{DROP_SENSOR} 제외(센서 17개, 이벤트 8개)": subset(S, N, ev_ids, nodes, drop_sensor=DROP_SENSOR),
        f"큰 사건 {DROP_EVENTS[0]}, {DROP_EVENTS[1]} 제외(센서 18개, 이벤트 6개)": subset(S, N, ev_ids, nodes, drop_events=DROP_EVENTS),
        f"둘 다 제외(센서 17개, 이벤트 6개)": subset(S, N, ev_ids, nodes, drop_sensor=DROP_SENSOR, drop_events=DROP_EVENTS),
    }
    res = {name: run_scenario(s[0], s[1]) for name, s in scen.items()}

    L = []
    w = L.append
    w("[서울시 강남구 센서 18개] T2 LOSO 민감도 (사후 분석, 판정 변경 없음, test 구간 미사용)")
    w("=" * 78)
    w("※ 타깃 기저선·규모 기지 가정 하의 상대 상승폭 추정이다(절대수위 추정 아님). 사전 판정(10/18 통과)은 바꾸지 않는다.")
    w("이벤트 복원추출 2000회(시드 0), 통계 = 센서 평균 이벤트 MAE(m), 95% CI.")
    for name, r in res.items():
        S_, N_, ev_, nd_ = scen[name]
        w("")
        w(f"[{name}] 이벤트 {ev_}, 센서 {len(nd_)}개")
        for m in SHOW:
            lo, hi = r["cis"][m]
            w(f"  {SHORT[m]:22s} {r['point'][r['k'][m]]:.4f}  [{lo:.4f}, {hi:.4f}]")
        w(f"  순위(낮은 오차 순, (1)~(5)): " + " < ".join(SHORT[m][:3] for m in r["rank"]))
        w(f"  (4)-(2): {r['d4_point']:+.4f} [{r['d4'][0]:+.4f}, {r['d4'][1]:+.4f}] → {verdict_text(*r['d4'])}")
        w(f"  (5)-(2): {r['d5_point']:+.4f} [{r['d5'][0]:+.4f}, {r['d5'][1]:+.4f}] → {verdict_text(*r['d5'])}")
        w(f"  사전 기준 통과 센서 수(점추정, 참고): {r['pass_point']}개 / {len(nd_)}개 (부트스트랩 95% 구간 {r['pass_ci']})")
    base_rank = res[next(iter(res))]["rank"]
    w("")
    w("순위 유지 여부 (기준 순위와 비교, (1)~(5) 5개 모델)")
    for name, r in res.items():
        w(f"  {name}: {'기준과 같음' if r['rank'] == base_rank else '기준과 다름'}  [{' < '.join(SHORT[m][:3] for m in r['rank'])}]")
    w("")
    w("주의: 사후 분석이다. 이벤트 6~8개의 부트스트랩이라 분포가 이산적이고 CI가 넓다. 제외는 결과를 본 뒤의 선택이므로 판정에 쓰지 않는다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
