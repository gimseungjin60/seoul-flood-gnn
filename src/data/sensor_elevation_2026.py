"""2026 pool 센서의 지표 고도(그래프 보완 실험 G1용, 규칙: src/analysis/t2_graph_rules.py, docs/14). 모델 학습 없음.

실행: python -m src.data.sensor_elevation_2026
입력: data/processed/gangnam2026_station_nodes.csv (좌표), Copernicus DEM GLO-30 타일 1개(없으면 내려받아 data/raw/dem/ 에 저장, git 제외)
출력: data/processed/gangnam2026_station_elevation.csv (node_id, role, lat, lon, elev_center_m, elev_m)

- elev_m = 센서 좌표가 들어 있는 화소를 중심으로 3x3 화소 최솟값. GLO-30은 건물·수목이 포함된 표면 고도(DSM)라서 도로 위 값을 고르려고 최솟값을 쓴다.
- 고도는 위치로 정해지는 값이며 수위 관측과 무관하다(엣지는 관측 없이 정의).
- DEM: Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
"""
import urllib.request

import pandas as pd
import rasterio

from src.analysis.t2_graph_rules import DEM_TILE, DEM_URL, DEM_WINDOW
from src.data.build_station_graph_2026 import OUT_NODES
from src.data.clean_nodes import PROCESSED
from src.data.raw_paths import RAW

DEM_PATH = RAW / "dem" / f"{DEM_TILE}.tif"
OUT_ELEV = PROCESSED / "gangnam2026_station_elevation.csv"


def ensure_dem():
    if not DEM_PATH.exists():
        DEM_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"DEM 내려받기: {DEM_URL}", flush=True)
        urllib.request.urlretrieve(DEM_URL, DEM_PATH)
    return DEM_PATH


def sample(src, lat: float, lon: float) -> tuple[float, float]:
    """(중심 화소 값, 3x3 최솟값)."""
    row, col = src.index(lon, lat)
    h = DEM_WINDOW // 2
    if not (h <= row < src.height - h and h <= col < src.width - h):
        raise ValueError(f"좌표가 DEM 타일 가장자리 밖이다: {lat}, {lon}")
    win = src.read(1, window=((row - h, row + h + 1), (col - h, col + h + 1))).astype(float)
    if src.nodata is not None and (win == src.nodata).any():
        raise ValueError(f"DEM 결측 화소: {lat}, {lon}")
    return float(win[h, h]), float(win.min())


def main() -> None:
    nodes = pd.read_csv(OUT_NODES)
    nodes = nodes[nodes.j2_pass & nodes.lat.notna()].reset_index(drop=True)
    with rasterio.open(ensure_dem()) as src:
        if src.crs.to_epsg() != 4326:
            raise ValueError(f"예상과 다른 좌표계: {src.crs}")
        vals = [sample(src, r.lat, r.lon) for r in nodes.itertuples()]
    out = nodes[["node_id", "role", "lat", "lon"]].copy()
    out["elev_center_m"] = [v[0] for v in vals]
    out["elev_m"] = [v[1] for v in vals]
    out.to_csv(OUT_ELEV, index=False, encoding="utf-8-sig", float_format="%.2f")
    print(f"저장: {OUT_ELEV.name} ({len(out)}개)")
    print(out.groupby("role")[["elev_m", "elev_center_m"]].describe().round(1).T.to_string())


if __name__ == "__main__":
    main()
