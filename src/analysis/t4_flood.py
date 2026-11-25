"""T4 실행·판정: 2025 침수흔적도의 실제 침수 피해와 하수관로 수위(은평·강서·서대문·도봉·강북)의 관계. 규칙은 src/analysis/t4_flood_rules.py 에 수위를 보기 전에 고정했다.

실행: python -m src.analysis.t4_flood      (먼저 fetch_sewer_api, geocode_sewer_api 가 끝나야 한다)
출력: data/processed/api2025_t4_report.txt, api2025_t4_ci.csv
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.analysis.t4_flood_rules import (BASE_DAYS, BIG_MM, BOOT_SEED, C_LOGIT, FINE_DATES, GU_CODE, H_BLOCKS, MAIN_DATES, MIN_DAY_COVER, MIN_GROUP,
                                         N_BOOT, R_MAIN_M, R_REF_M, ap_verdict, auc_verdict)
from src.data.clean_nodes import PROCESSED
from src.data.fetch_sewer_api import OUT_DIR
from src.data.raw_paths import FLOOD_TRACE_DIR

GEOCODE = PROCESSED / "api2025_flood_gu_geocoding.csv"
OUT_REPORT = PROCESSED / "api2025_t4_report.txt"
OUT_CI = PROCESSED / "api2025_t4_ci.csv"
ERR_M = -0.99                       # 오류 코드 -1
BLK = pd.Timedelta(minutes=10)
PER_DAY = 144


# ----------------------------------------------------------------------------- 자료
def load_levels(ids: list[str]) -> pd.DataFrame:
    """(10분 블록 시작 시각 x 센서) 평균 수위(m). 오류값 제외."""
    parts = []
    for f in sorted(OUT_DIR.glob("*.csv")):
        d = pd.read_csv(f, usecols=["UNQ_NO", "MSRMT_YMD", "MSRMT_WATL"], dtype={"UNQ_NO": str})
        parts.append(d[d.UNQ_NO.isin(ids)])
    d = pd.concat(parts, ignore_index=True)
    d["t"] = pd.to_datetime(d.MSRMT_YMD.str.slice(0, 19)).dt.floor("10min")
    d["lv"] = pd.to_numeric(d.MSRMT_WATL, errors="coerce")
    d.loc[~(d.lv > ERR_M), "lv"] = np.nan
    p = d.pivot_table(index="t", columns="UNQ_NO", values="lv", aggfunc="mean")
    return p.reindex(pd.date_range(p.index.min().normalize(), p.index.max().normalize() + pd.Timedelta(days=1) - BLK, freq="10min"))


def load_damage() -> gpd.GeoDataFrame:
    p = next(p for p in FLOOD_TRACE_DIR.rglob("*.shp") if p.parent.name.startswith("2025"))
    for enc in ("utf-8", "cp949"):
        try:
            g = gpd.read_file(p, encoding=enc)
            break
        except Exception:
            g = None
    s = g["피해일시"].astype(str)
    g["date"] = s.str.extract(r"^(\d{8})")[0]
    hh = s.str.extract(r"-\s*(\d{3,4})\s*$")[0].str.zfill(4)
    g["when"] = pd.to_datetime(g.date + hh, format="%Y%m%d%H%M", errors="coerce")
    g["gu"] = g.ADM_CD.astype("int64").astype(str).str[:5]
    g = g[g.gu.isin(GU_CODE)].copy()
    return gpd.GeoDataFrame(g[["date", "when", "gu"]], geometry=g.geometry.centroid, crs=g.crs)


# ----------------------------------------------------------------------------- 분석
def auc_boot(score: np.ndarray, y: np.ndarray, rng) -> tuple[float, float, float]:
    point = roc_auc_score(y, score)
    n, b = len(y), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        b.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def features(a: pd.DataFrame):
    """a (블록 x 센서) mm. 기점 t = 블록 끝. 특징과 점수(t 까지의 정보만). 반환: dict of (B, J) 배열."""
    A = a.to_numpy()
    B, J = A.shape
    out = {"a": A}
    miss = np.zeros((B, J))
    diffs = []
    for k in (1, 3, 6):
        lag = np.full((B, J), np.nan)
        lag[k:] = A[:-k]
        d = A - lag
        miss += ~np.isfinite(d)
        diffs.append(np.nan_to_num(d) / 100)
        out[f"d{k}"] = d
    mx = pd.DataFrame(A).rolling(6, min_periods=1).max().to_numpy()
    X = np.stack([np.nan_to_num(A) / 100, *diffs, miss, np.nan_to_num(mx) / 100], axis=-1)
    out["X"] = X
    out["S0"] = A
    out["S1"] = A + np.where(np.isfinite(out["d3"]), out["d3"], 0.0)
    return out


def main() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    sen = pd.read_csv(GEOCODE, dtype={"node_id": str})
    sen = sen[sen.geocoding_status == "OK"].reset_index(drop=True)
    lv = load_levels(sen.node_id.tolist())
    sen = sen[sen.node_id.isin(lv.columns)].reset_index(drop=True)
    sp = gpd.GeoDataFrame(sen[["node_id", "district_name"]], geometry=[Point(x, y) for x, y in zip(sen.longitude, sen.latitude)], crs=4326).to_crs(5179)
    dmg = load_damage()
    # 기준 수위(인과적: 08-10~08-12 중앙값). 기준 관측이 없는 센서는 제외.
    base_mask = lv.index.strftime("%Y%m%d").isin(BASE_DAYS)
    base = lv[base_mask].median()
    keep = base.dropna().index.tolist()
    lv, sp = lv[keep], sp[sp.node_id.isin(keep)].reset_index(drop=True)
    ids = sp.node_id.tolist()
    a = (lv[ids] - base[ids]) * 1000
    day = a.index.strftime("%Y%m%d")
    # 유효 센서-일
    cover = a.notna().groupby(day).mean()
    valid = (cover >= MIN_DAY_COVER)
    L = ["[2025 침수 피해와 하수관로 수위] 은평·강서·서대문·도도봉·강북 (규칙은 수위 보기 전 고정: src/analysis/t4_flood_rules.py)".replace("도도봉", "도봉"), "=" * 80,
         f"센서 {len(ids)}개(좌표 확보, 기준 수위 있음): " + ", ".join(f"{k} {v}" for k, v in sp.district_name.value_counts().items()) + f". 자료 {a.index.min():%m-%d} ~ {a.index.max():%m-%d}.",
         "피해는 풍수해보험금·재난지원금 기록(하수관 수위와 무관한 피해 포함 가능). 탐색적 결과이며 사건이 2~3개뿐이다.", ""]
    csv = []
    near = {}                                                    # R -> (센서 x 피해) 거리 행렬로 쓰는 대신 센서별 인덱스
    dist = np.array([[p.distance(q) for q in dmg.geometry] for p in sp.geometry])        # (센서, 피해) m
    dd = dmg.date.to_numpy()

    # ---- 분석 A: 날짜 단위 연관
    L.append(f"[분석 A: 연관] (센서, 날짜): 점수 = 그 날 최대 상승폭, 라벨 = 반경 {R_MAIN_M} m 안 그 날 피해 유무. ROC-AUC, 센서 부트스트랩 95% 구간")
    for tag, dates in (("주", MAIN_DATES), ("참고", tuple(d for d in ("20250814", "20250821", "20250824") if d in valid.index))):
        for D in dates:
            if D not in valid.index:
                L.append(f"  {D}: 자료 없음")
                continue
            rows = []
            for j, n in enumerate(ids):
                if not valid.loc[D, n]:
                    continue
                r = {"node": n, "score": float(a.loc[day == D, n].max())}
                for R in (R_MAIN_M, *R_REF_M):
                    r[f"y{R}"] = int(((dist[j] <= R) & (dd == D)).any())
                rows.append(r)
            t = pd.DataFrame(rows)
            y = t[f"y{R_MAIN_M}"].to_numpy()
            if min(y.sum(), len(y) - y.sum()) < MIN_GROUP:
                L.append(f"  [{tag}] {D}: 유효 센서 {len(t)}개, 양성 {int(y.sum())}개 → 표본 부족")
                continue
            auc, lo, hi = auc_boot(t.score.to_numpy(), y, rng)
            v = auc_verdict(lo, hi)
            ref = ", ".join(f"R={R}m: AUC {roc_auc_score(t[f'y{R}'], t.score):.2f}" for R in R_REF_M if 0 < t[f"y{R}"].sum() < len(t))
            L.append(f"  [{tag}] {D}: 유효 센서 {len(t)}개, 근처 피해 {int(y.sum())}개 | AUC {auc:.2f} [{lo:.2f}, {hi:.2f}] → {v}  (참고 {ref})")
            csv.append({"stat": f"A_{D}_auc_R{R_MAIN_M}", "point": auc, "lo": lo, "hi": hi})
    L.append("")

    # ---- 분석 B: 선행 시간
    L.append(f"[분석 B: 선행 시간] 시각 단위 날짜(08-21~24)의 (센서, 피해) 쌍(반경 {R_MAIN_M} m): 선행 = 피해 시각 − 그 날 처음 상승폭 {BIG_MM:.0f} mm 이상인 시각 (기술, 판정 없음)")
    leads, notreached, total = [], 0, 0
    for D in FINE_DATES:
        if D not in valid.index:
            continue
        for j, n in enumerate(ids):
            if not valid.loc[D, n]:
                continue
            idx = np.flatnonzero((dist[j] <= R_MAIN_M) & (dd == D) & dmg.when.notna().to_numpy())
            if not len(idx):
                continue
            s = a.loc[day == D, n]
            hit = s[s >= BIG_MM]
            for i in idx:
                total += 1
                if hit.empty:
                    notreached += 1
                else:
                    leads.append((dmg.when.iloc[i] - (hit.index[0] + BLK)).total_seconds() / 60)
    if leads:
        q = np.percentile(leads, [25, 50, 75])
        L.append(f"  쌍 {total}개 중 그 날 {BIG_MM:.0f} mm 에 도달하지 않은 쌍 {notreached}개, 도달한 쌍 {len(leads)}개. 도달한 쌍의 선행(분) 분위수 25/50/75 = {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}, 선행 > 0 비율 {np.mean(np.array(leads) > 0):.0%}")
    else:
        L.append(f"  쌍 {total}개, 도달한 쌍 없음")
    L.append("")

    # ---- 분석 C: 예측 모델 (leave-one-day-out)
    F = features(a)
    B, J = a.shape
    fine_blocks = np.flatnonzero(np.isin(day, FINE_DATES))
    y = np.zeros((B, J))
    okmask = np.zeros((B, J), bool)
    wh = dmg.when.to_numpy("datetime64[ns]")
    fine_dmg = np.flatnonzero(np.isin(dd, FINE_DATES) & dmg.when.notna().to_numpy())
    tend = (a.index + BLK).to_numpy("datetime64[ns]")
    for j in range(J):
        di = fine_dmg[dist[j, fine_dmg] <= R_MAIN_M]
        times = np.sort(wh[di])
        for b in fine_blocks:
            if not (valid.loc[day[b], ids[j]] and np.isfinite(F["a"][b, j]) and F["a"][b, j] < BIG_MM):
                continue
            okmask[b, j] = True
            lo_i, hi_i = np.searchsorted(times, tend[b], "right"), np.searchsorted(times, tend[b] + H_BLOCKS * BLK.to_timedelta64(), "right")
            y[b, j] = 1.0 if hi_i > lo_i else 0.0
    rows = []
    for D in FINE_DATES:
        test_b = np.array([b for b in fine_blocks if day[b] == D])
        train_b = np.array([b for b in fine_blocks if day[b] != D])
        tr, te = okmask[train_b], okmask[test_b]
        if not te.any() or y[train_b][tr].sum() == 0 or y[test_b][te].sum() == 0:
            continue
        sc = StandardScaler().fit(F["X"][train_b][tr])
        clf = LogisticRegression(C=C_LOGIT, class_weight="balanced", max_iter=1000).fit(sc.transform(F["X"][train_b][tr]), y[train_b][tr])
        bb, jj = np.nonzero(te)
        rows.append(pd.DataFrame({"day": D, "j": jj, "y": y[test_b][te], "L": clf.predict_proba(sc.transform(F["X"][test_b][te]))[:, 1],
                                  "S0": F["S0"][test_b][te], "S1": F["S1"][test_b][te]}))
    L.append(f"[분석 C: 예측 모델] 기점 상승폭 {BIG_MM:.0f} mm 미만인 (시각, 센서), 이후 3시간 안 반경 {R_MAIN_M} m 시각 단위 피해(08-21~24). 날짜 단위 leave-one-day-out, 로지스틱 회귀 하나")
    if rows:
        df = pd.concat(rows, ignore_index=True)
        L.append(f"  평가 표본 {len(df):,}개, 양성 {int(df.y.sum()):,}개 (기준 양성 비율 {df.y.mean():.1%}), 양성이 있는 센서 {df[df.y == 1].j.nunique()}개 / {df.j.nunique()}개")
        ap = {m: average_precision_score(df.y, df[m]) for m in ("S0", "S1", "L")}
        L.append("  평균정밀도(AP): " + ", ".join(f"{m} {v:.3f}" for m, v in ap.items()) + "   (S0 현재 상승폭, S1 추세 외삽, L 로지스틱)")
        groups = {j: g for j, g in df.groupby("j")}
        sensors = np.array(list(groups))
        verd = {}
        for base_m in ("S0", "S1"):
            d = []
            for _ in range(N_BOOT):
                s = pd.concat([groups[j] for j in rng.choice(sensors, len(sensors))], ignore_index=True)
                if s.y.sum() == 0:
                    continue
                d.append(average_precision_score(s.y, s.L) - average_precision_score(s.y, s[base_m]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            pt = ap["L"] - ap[base_m]
            verd[base_m] = ap_verdict(lo, hi)
            L.append(f"  AP(L) − AP({base_m}) = {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] → {verd[base_m]}")
            csv.append({"stat": f"C_L_minus_{base_m}", "point": pt, "lo": float(lo), "hi": float(hi)})
        ok = verd["S0"] == "L이 낫다" and verd["S1"] == "L이 낫다"
        L += ["", "[결론 규칙] 두 비교가 모두 'L이 낫다'여야 '침수 피해 예측이 된다'고 말한다.",
              "  → " + ("수위 기반 모델이 단순 기준보다 침수 피해를 먼저 가려냄(탐색적, 사건 4일)." if ok else "단순 기준을 확실히 넘지 못했다. '침수 피해 예측이 된다'고 보고하지 않는다.")]
    else:
        L.append("  평가 가능한 표본이 없다.")
    L += ["", "※ 사건 2~3개, 한 해, 탐색적. 강우 자료 없음. 피해 기록은 수위와 무관한 피해를 포함할 수 있다. 좌표는 주소 지오코딩이라 오차가 있다."]
    pd.DataFrame(csv).to_csv(OUT_CI, index=False, encoding="utf-8-sig", float_format="%.5f")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
