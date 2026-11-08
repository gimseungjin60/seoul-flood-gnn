"""대시보드 표시용 '평소 대비 상승 단계' 설계 근거 (docs/16). 판정이 아니라 설계용 기술 통계다. 모델 학습 없음.

실행: python -m src.analysis.rise_levels   (2026 부분은 t2_2026 --part loso / gnn 예측 파일이 있어야 한다)
출력: data/processed/rise_levels_report.txt

- 상승폭 = 10분 블록 수위 − 그 센서의 요일x시 중앙값(기간 전체). 단위 mm. 관로 제원이 없어 '위험'이 아니라 '평소 대비 얼마나 올랐나'만 나타낸다.
- 단계 경계(STAGE_MM)는 2025 자료(센서 18개, train+val, 이벤트 8개)로 정했다:
  50 mm = 측정 단위 10 mm의 5배(P3 대상 기준과 같음), 200 mm ≈ 2025 이벤트 첨두 상승 중앙값(201), 500 mm ≈ 2025 이벤트 첨두 상위 25%(523).
- 2026(평가 대상 17개)은 확인용이다. 이웃 전용 센서는 이벤트 밖 고수위 비율로 기록 안정성만 본다.
- 센서 없는 지점 추정(IDW·GNN·선형)이 단계를 얼마나 맞히는지는 2026 재비교 예측(같은 평가 틀)으로 본다. 그 예측의 상승폭은 폴드 학습 구간의
  요일x시 평균 기준이라 위 정의와 기저선이 조금 다르다(단계 일치 확인용으로만 쓴다).
"""
import numpy as np
import pandas as pd

from src.analysis.spatial_signal_profile_adjusted import week_profile
from src.analysis.t2_2026 import OUT_PRED, build_inputs as inputs_2026, load_roles
from src.analysis.t2_loso import build_inputs as inputs_2025, event_blocks
from src.data.clean_nodes import PROCESSED
from src.data.prepare_rainfall_2026 import OUT_EVENTS

STAGE_MM = (50, 200, 500)
STAGE_NAMES = ("평상", "상승", "큰 상승", "매우 큰 상승")
UNSTABLE_NONEVENT_SHARE = 0.02          # 이벤트 밖 블록 중 100 mm 이상이 2% 넘으면 '기록 불안정'
OUT_REPORT = PROCESSED / "rise_levels_report.txt"


def stage(x_mm) -> np.ndarray:
    return np.digitize(x_mm, STAGE_MM)


def rise_mm(inp: dict) -> np.ndarray:
    prof = week_profile(inp["blk"], inp["how"], np.ones(len(inp["blk"]), bool), "median")
    return (inp["blk"] - prof[inp["how"]]) * 1000


def event_mask(bt, events):
    ev = event_blocks(bt, events)
    m = np.zeros(len(bt), bool)
    for _, idx in ev:
        m[idx] = True
    return ev, m


def peaks(r: np.ndarray, ev) -> np.ndarray:
    """(센서, 이벤트) 첨두 상승. 이벤트 창 안 유효 블록 6개 미만은 제외."""
    return np.array([np.nanmax(r[idx, j]) for _, idx in ev for j in range(r.shape[1]) if np.isfinite(r[idx, j]).sum() >= 6])


def describe(name: str, r: np.ndarray, ev, inev) -> list[str]:
    ok = np.isfinite(r)
    pk = peaks(r, ev)
    L = [f"\n[{name}] 센서 {r.shape[1]}개, 이벤트 {len(ev)}개, (센서, 이벤트) 첨두 {len(pk)}개",
         "  이벤트 첨두 상승(mm) 백분위 10/25/50/75/90: " + ", ".join(f"{v:.0f}" for v in np.percentile(pk, [10, 25, 50, 75, 90])),
         "  첨두 단계 분포: " + ", ".join(f"{n} {c / len(pk):.0%}" for n, c in zip(STAGE_NAMES, np.bincount(stage(pk), minlength=4)))]
    out = ok & ~inev[:, None]
    L.append("  이벤트 밖 블록 단계 분포: " + ", ".join(f"{n} {c / out.sum():.2%}"
                                              for n, c in zip(STAGE_NAMES, np.bincount(stage(r[out]), minlength=4))))
    return L


def estimate_agreement() -> list[str]:
    z, g = np.load(OUT_PRED["loso"], allow_pickle=True), np.load(OUT_PRED["gnn"], allow_pickle=True)
    bt = pd.DatetimeIndex(z["bt"])
    act = z["actual_m"].astype(float) * 1000
    preds = {"IDW": z["pred_idw_p1"], "강우+이웃(선형)": z["pred_ridge_rain_time_nbr"], "GNN": g["pred_gnn"]}
    ev, _ = event_mask(bt, inputs_2026_events())
    ok = np.isfinite(act) & np.isfinite(z["pred_idw_p1"])
    L = ["\n[센서 없는 지점 추정이 단계를 맞히는가] 2026 평가 대상 17개를 하나씩 가리고 추정(2026 재비교 예측), 이벤트 첨두 기준"]
    for name, p in preds.items():
        p = p.astype(float) * 1000
        A, P = [], []
        for _, idx in ev:
            for j in range(act.shape[1]):
                ii = idx[ok[idx, j]]
                if len(ii) >= 6:
                    A.append(np.nanmax(act[ii, j]))
                    P.append(np.nanmax(p[ii, j]))
        A, P = stage(np.array(A)), stage(np.array(P))
        hi = A >= 2
        L.append(f"  {name}: 단계 일치 {np.mean(A == P):.0%}, 한 단계 이내 {np.mean(abs(A - P) <= 1):.0%}, 낮게 추정 {np.mean(P < A):.0%}, "
                 f"높게 추정 {np.mean(P > A):.0%} | 실제 '큰 상승' 이상 {hi.sum()}개 중 추정도 '큰 상승' 이상 {np.mean(P[hi] >= 2):.0%}, "
                 f"실제 '매우 큰 상승' {(A == 3).sum()}개 중 추정도 '매우 큰 상승' {np.mean(P[A == 3] == 3):.0%}")
    return L


def unstable_nodes(r: np.ndarray, inev: np.ndarray, cols) -> list[tuple[int, float]]:
    """기록 불안정 센서: 이벤트 밖 블록 중 상승 100 mm 이상 비율이 UNSTABLE_NONEVENT_SHARE 초과. 반환 [(열, 비율)]."""
    out = []
    for c in cols:
        ok = np.isfinite(r[:, c]) & ~inev
        share = float(np.mean(r[ok, c] >= 100))
        if share > UNSTABLE_NONEVENT_SHARE:
            out.append((c, share))
    return out


def inputs_2026_events():
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    return ev[ev.is_eval].reset_index(drop=True)


def main() -> None:
    L = ["[강남] 평소 대비 상승 단계 설계 근거 (docs/16). 판정 아님, 설계용 기술 통계.",
         f"단계: {STAGE_NAMES[0]} < {STAGE_MM[0]} mm <= {STAGE_NAMES[1]} < {STAGE_MM[1]} mm <= {STAGE_NAMES[2]} < {STAGE_MM[2]} mm <= {STAGE_NAMES[3]}",
         "상승폭 = 10분 수위 − 센서별 요일x시 중앙값. 관로 크기를 모르므로 위험도가 아니다."]
    i25 = inputs_2025()
    ev25, in25 = event_mask(i25["bt"], i25["events"])
    L += describe("2025 센서 18개 (단계 경계를 정한 자료)", rise_mm(i25), ev25, in25)

    roles = load_roles()
    i26 = inputs_2026(roles["pool"])
    r26 = rise_mm(i26)
    ev26, in26 = event_mask(i26["bt"], i26["events"])
    tg = [roles["pool"].index(n) for n in roles["targets"]]
    nb = [c for c in range(len(roles["pool"])) if c not in tg]
    L += describe("2026 평가 대상 17개 (확인용)", r26[:, tg], ev26, in26)
    L.append("\n[2026 이웃 전용 센서 기록 안정성] 이벤트 밖 블록 중 100 mm 이상 비율 "
             f"(> {UNSTABLE_NONEVENT_SHARE:.0%}면 '기록 불안정', 단계 색 없이 회색 표시)")
    unstable = [f"{roles['pool'][c]} {share:.0%}" for c, share in unstable_nodes(r26, in26, nb)]
    L.append(f"  불안정 {len(unstable)}/{len(nb)}개: " + ", ".join(unstable))
    L += estimate_agreement()
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
