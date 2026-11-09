"""센서 위치와 서울시 풍수해 침수예상도(shp, 침수심 구간별 면) 겹치기, 그리고 침수예상 영역 안/밖 센서의 강우 반응 비교.
규칙은 결과(집단별 반응)를 보기 전에 이 docstring 으로 고정했다(2026-11-09). 판정은 하나뿐이고 나머지는 참고다.

자료: data/raw/flood/침수예상도/DS_FLOODING_{1..6}.shp (git 제외). 레이어 이름에 침수심 구간이 있다: 1 = '침수심1'(구간 값은 이름에 없음, 2번이 0.5~1.0 m라 0.5 m 미만으로 추정),
  2 = 0.5~1.0 m, 3 = 1.0~1.5 m, 4 = 1.5~2.0 m, 5 = 2.0~3.0 m, 6 = 3.0 m 이상. 파일 수정일 2023-02-15(열린데이터광장 API floodingDs 의 2026-11-03 갱신본과
  다른 판일 수 있음, API 에는 좌표가 없어 연결 못 함). .prj 가 없어 좌표계는 EPSG:5186 으로 가정(열린데이터광장 설명, 센서 좌표와 겹침으로 타당성 확인).
  CAD 변환 도형이라 일부 도형이 유효하지 않아 shapely.make_valid 로 보정했다.
겹치기: 센서 위경도 → EPSG:5186. 센서별 '영역 안에 있는 가장 깊은 구간'(0 = 어느 구간에도 안 들어감)과 가장 가까운 면까지의 거리(m).
판정(주): docs/21 의 기록 안정 센서 39개를 침수예상도 안(구간 1 이상) / 밖으로 나누고, 센서별 '총 50 mm 이상 비 5건에서의 최대 상승 중앙값(heavy_median_mm)'의
  두 집단 중앙값 차이(안 − 밖)를 센서 단위 부트스트랩(집단별 복원추출, 2000회, 시드 0) 95% 구간으로 본다. 구간이 0을 포함하면 "구분 불가", 하한 > 0 이면 "안쪽 센서가 더 크게 오름",
  상한 < 0 이면 "안쪽 센서가 덜 오름". 센서 수가 집단마다 5 미만이면 "표본 부족".
참고(판정 아님): 가장 가까운 면까지 거리와 heavy_median_mm 의 순위상관, 집단별 개수.
한계: 침수예상도는 지표면 침수 시뮬레이션이고 하수관 수위와 같은 것이 아니다. 하천 범람·반지하 침수는 수위와 무관할 수 있다. 침수심 2 이상 영역에는 센서가 없다.
  침수심1 영역이 넓어(58 km², 서울의 약 10%) 안/밖 구분이 약한 위험 정보다. 센서 수가 적다.

실행: python -m src.analysis.flood_overlay
출력: data/processed/gangnam2026_sensor_flood_overlay.csv, gangnam2026_flood_overlay_report.txt
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from shapely import make_valid
from shapely.geometry import Point

from src.analysis.rain_response import OUT_SENSOR
from src.data.clean_nodes import PROCESSED
from src.data.raw_paths import FLOOD_FORECAST_DIR

SHP_DIR = FLOOD_FORECAST_DIR
OUT_OVERLAY = PROCESSED / "gangnam2026_sensor_flood_overlay.csv"
OUT_REPORT = PROCESSED / "gangnam2026_flood_overlay_report.txt"
DEPTH_NAME = {0: "예상 침수 영역 밖", 1: "침수심1(0.5 m 미만 추정)", 2: "0.5~1.0 m", 3: "1.0~1.5 m", 4: "1.5~2.0 m", 5: "2.0~3.0 m", 6: "3.0 m 이상"}
N_BOOT, BOOT_SEED, MIN_GROUP = 2000, 0, 5


def load_layers() -> dict:
    out = {}
    for k in range(1, 7):
        g = gpd.read_file(SHP_DIR / f"DS_FLOODING_{k}.shp", encoding="cp949")
        out[k] = [v for v in (make_valid(x) for x in g.geometry) if not v.is_empty]
    return out


def overlay(nodes: pd.DataFrame, layers: dict) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(nodes[["node_id"]].copy(), geometry=[Point(x, y) for x, y in zip(nodes.lon, nodes.lat)], crs=4326).to_crs(5186)
    rows = []
    for p in pts.itertuples():
        inside = [k for k in range(1, 7) if any(x.contains(p.geometry) for x in layers[k])]
        dist = [d for d in (x.distance(p.geometry) for k in layers for x in layers[k]) if np.isfinite(d)]
        rows.append(dict(node_id=p.node_id, flood_depth_class=max(inside) if inside else 0, dist_to_flood_m=round(min(dist)) if dist else np.nan))
    return pd.DataFrame(rows)


def boot_median_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(N_BOOT)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.median(a) - np.median(b)), float(lo), float(hi)


def main() -> None:
    nodes = pd.read_csv(PROCESSED / "gangnam2026_station_nodes.csv", encoding="utf-8-sig")
    ov = overlay(nodes, load_layers())
    ov.to_csv(OUT_OVERLAY, index=False, encoding="utf-8-sig")
    resp = pd.read_csv(OUT_SENSOR).merge(ov, on="node_id")
    resp["inside"] = resp.flood_depth_class >= 1
    a, b = resp[resp.inside].heavy_median_mm.dropna().to_numpy(), resp[~resp.inside].heavy_median_mm.dropna().to_numpy()
    L = ["[강남 2026] 센서 위치와 풍수해 침수예상도(침수심 구간별 shp) 겹치기 (규칙은 결과 보기 전 고정: src/analysis/flood_overlay.py)", "=" * 78,
         "침수예상도는 지표면 침수 시뮬레이션이며 하수관 수위와 다르다. 파일은 2023-02-15판, 좌표계 EPSG:5186 가정.", "",
         "[센서 위치별 침수예상도 구간] (연구에 쓰는 센서 48개 + 제외 센서 6개 = 54개)"]
    for k, n in ov.flood_depth_class.value_counts().sort_index().items():
        L.append(f"  {DEPTH_NAME[k]:<24} {n}개")
    out = ov[ov.flood_depth_class == 0]
    L += [f"  영역 밖 센서의 가장 가까운 예상 면까지 거리: 중앙 {out.dist_to_flood_m.median():.0f} m, 최대 {out.dist_to_flood_m.max():.0f} m", "",
          f"[판정] 기록 안정 센서 {len(resp)}개: 예상 침수 영역 안 {len(a)}개, 밖 {len(b)}개 (50 mm 이상 비 5건에서의 최대 상승 중앙값, mm)"]
    if min(len(a), len(b)) < MIN_GROUP:
        L.append("  → 표본 부족")
    else:
        d, lo, hi = boot_median_diff(a, b)
        v = "구분 불가" if lo <= 0 <= hi else ("안쪽 센서가 더 크게 오름" if lo > 0 else "안쪽 센서가 덜 오름")
        L.append(f"  집단 중앙값: 안 {np.median(a):.0f} mm, 밖 {np.median(b):.0f} mm. 차이(안 − 밖) {d:+.0f} mm [{lo:+.0f}, {hi:+.0f}] → {v}")
    ok = resp.dist_to_flood_m.notna() & resp.heavy_median_mm.notna()
    rho, p = spearmanr(resp.loc[ok, "dist_to_flood_m"], resp.loc[ok, "heavy_median_mm"])
    L += ["", f"[참고] 가장 가까운 예상 침수 면까지 거리와 최대 상승 중앙값의 순위상관 {rho:+.2f} (p={p:.2f}, 센서 {int(ok.sum())}개). 판정 아님.",
          "", "※ 침수심 2 이상 영역에는 센서가 없고, 침수심1 영역이 넓어 안/밖 구분이 약하다. 센서가 적어 구간이 넓다.",
          "※ 위험 판정이 아니라 침수 예상 영역과의 위치 관계일 뿐이다."]
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
