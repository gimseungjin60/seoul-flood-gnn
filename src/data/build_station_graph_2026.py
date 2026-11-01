"""2026 이웃 후보(pool) 센서 좌표 확보 → 도로망 연결 → 센서 간 도로거리 + k=5 이웃 관측률 진단 (모델 학습 없음).

실행: python -m src.data.build_station_graph_2026            # 결과 파일에 없는 센서만 Kakao 조회(재실행해도 재호출 안 함)
입력(읽기만): data/processed/gangnam2026_sensor_selection.csv, gangnam2026_station_timeseries.npz,
             gangnam_geocoding_result.csv(2025 좌표, 위치 문구가 같으면 재사용)
출력: data/processed/gangnam2026_geocoding_result.csv, gangnam2026_station_nodes.csv,
      gangnam2026_station_pairwise_road_distance.csv, gangnam2026_station_graph_report.txt
      data/processed/gangnam2026_road_network.graphml (OSM은 시간에 따라 바뀌므로 재현성을 위해 커밋, 2025 graphml과 같음)

- 좌표를 추측하지 않는다(2025 geocoder.decide 규칙 그대로: 결과 1건 + 강남구일 때만). 시설명 수동 재검색은 하지 않는다.
- J2 좌표 검증 통과(docs/11): 좌표 있음 + 강남 범위 사각형 안 + 매칭 주소 강남구 + 최근접 도로 노드 100 m 이내(CANDIDATE).
  근접·중복·고립 경고는 검토 표시일 뿐 제외 사유가 아니다(2025와 같음).
- 거리 = 무방향 도로망 최단거리(2025 build_station_graph.pairwise_road_distance). 실제 하수관로 연결이 아니다.
"""
import os
import sys

import numpy as np
import osmnx as ox
import pandas as pd
from dotenv import load_dotenv

from src.analysis.t2_2026_rules import GNN_K, jump_excluded, sensor_roles
from src.data.assign_rain_gauges import node_blocks
from src.data.build_road_network import fetch_graph, nearest_edge_dist, snap_to_nodes, study_bbox
from src.data.build_station_graph import pairwise_road_distance
from src.data.clean_nodes import PROCESSED
from src.data.geocoder import OUT_CSV as GEO_2025, geocode
from src.data.inspect_water_level import PROJECT_ROOT
from src.data.prepare_station_timeseries_2026 import OUT_NPZ as TS_NPZ, OUT_SELECTION
from src.data.validate_coordinates import GANGNAM_BBOX, check_coordinates

OUT_GEO = PROCESSED / "gangnam2026_geocoding_result.csv"
OUT_NODES = PROCESSED / "gangnam2026_station_nodes.csv"
OUT_PAIRS = PROCESSED / "gangnam2026_station_pairwise_road_distance.csv"
OUT_GRAPHML = PROCESSED / "gangnam2026_road_network.graphml"
REPORT = PROCESSED / "gangnam2026_station_graph_report.txt"
REUSED = "REUSED_2025"
ROAD_BBOX = study_bbox(pd.Series(GANGNAM_BBOX["lat"]), pd.Series(GANGNAM_BBOX["lon"]))  # (W, S, E, N) + 500 m


def geocoding_table(pool: pd.DataFrame, geo25: pd.DataFrame) -> pd.DataFrame:
    """pool 센서 → 지오코딩 입력. 2025에 OK였고 위치 문구가 같으면 좌표 재사용, 나머지는 PENDING."""
    d = pool[["node_id", "location"]].rename(columns={"location": "search_keyword"}).assign(
        district_code="23", district_name="강남", location_source="2026_original_location",
        geocoding_status="PENDING", geocoding_method=None, latitude=np.nan, longitude=np.nan,
        n_results=np.nan, matched_address=None, location_basis="ADDRESS")
    old = geo25.set_index("node_id")
    for i, r in d.iterrows():
        if r.node_id in old.index:
            o = old.loc[r.node_id]
            if o.geocoding_status == "OK" and o.search_keyword == r.search_keyword:
                d.loc[i, ["geocoding_status", "geocoding_method", "latitude", "longitude", "n_results",
                          "matched_address", "location_basis"]] = \
                    ["OK", REUSED, o.latitude, o.longitude, o.n_results, o.matched_address, o.location_basis]
    return d.reset_index(drop=True)


def run_geocoding(table: pd.DataFrame) -> pd.DataFrame:
    """이미 결과 파일에 있는 센서는 그 결과를 쓰고, 없는 센서만 API 호출."""
    if OUT_GEO.exists():
        prev = pd.read_csv(OUT_GEO, dtype={"district_code": str})
        prev = prev[prev.geocoding_method != REUSED].set_index("node_id")   # 재사용 행은 항상 2025 파일에서 다시 만든다
        done = table["node_id"].isin(prev.index) & (table["geocoding_status"] == "PENDING")
        table = pd.concat([prev.loc[table.loc[done, "node_id"]].reset_index(), table[~done]], ignore_index=True)
    pending = table["geocoding_status"] == "PENDING"
    if pending.any():
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.getenv("KAKAO_REST_API_KEY")
        if not key:
            sys.exit(".env에 KAKAO_REST_API_KEY가 없습니다.")
        table = pd.concat([table[~pending], geocode(table[pending], key)], ignore_index=True)  # geocode는 주소 열을 비우므로 대상 행만
    return table.sort_values("node_id").reset_index(drop=True)


def road_graph():
    """강남 범위 사각형(validate_coordinates.GANGNAM_BBOX) + 500 m 고정 범위. 받은 graphml이 있으면 재사용(Overpass 재요청 안 함)."""
    if OUT_GRAPHML.exists() and OUT_GRAPHML.stat().st_mtime > 0:
        G = ox.load_graphml(OUT_GRAPHML)
        if G.graph.get("bbox_2026") == str(ROAD_BBOX):
            return G
    G = fetch_graph(ROAD_BBOX)
    G.graph["bbox_2026"] = str(ROAD_BBOX)
    ox.save_graphml(G, OUT_GRAPHML)
    return G


def neighbor_availability(ts_path, nodes: pd.DataFrame, pairs: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """타깃별: k=5 최근접 이웃(거리 고정)의 10분 블록 관측률과, 블록당 관측된 k=5 이웃 수 / IDW 후보 수."""
    z = np.load(ts_path)
    ids = list(z["node_ids"])
    blk = node_blocks(z["values"], z["mask"])
    obs = pd.DataFrame(np.isfinite(blk), columns=ids)
    sym = pd.concat([pairs, pairs.rename(columns={"node_a": "node_b", "node_b": "node_a"})])
    rows = []
    for t in roles["targets"]:
        cand = [n for n in roles["pool"] if n != t]
        dist = sym[(sym.node_a == t) & sym.node_b.isin(cand)].sort_values("road_dist_m")
        k5 = dist.node_b.head(GNN_K).tolist()
        rows.append({"target": t, "k5": ", ".join(k5), "k5_max_dist_m": dist.road_dist_m.head(GNN_K).max(),
                     "k5_block_obs_rate": float(obs[k5].to_numpy().mean()),
                     "k5_observed_per_block": float(obs[k5].sum(axis=1).mean()),
                     "k5_none_observed_share": float((obs[k5].sum(axis=1) == 0).mean()),
                     "idw_observed_per_block": float(obs[cand].sum(axis=1).mean())})
    return pd.DataFrame(rows)


def build_report(geo, chk, link, nodes, pairs, roles, avail) -> str:
    L = []
    w = L.append
    w("[강남 2026] pool 센서 좌표·도로거리·k=5 이웃 관측률 (모델 학습 없음)")
    w("=" * 66)
    w("※ 거리는 도로망 최단거리 근사이며 실제 하수관로 연결이 아니다.")
    w(f"\n1. 지오코딩 (pool {len(geo)}개)")
    w("  " + ", ".join(f"{k} {v}" for k, v in geo.groupby(["geocoding_status", "geocoding_method"], dropna=False)
                       .size().items()))
    w(f"  좌표 기준: {geo.location_basis.value_counts().to_dict()}")
    for r in geo[geo.geocoding_status != "OK"].itertuples():
        w(f"  {r.node_id} {r.geocoding_status}: {r.search_keyword}")
    w("\n2. 좌표 검증 (J2)")
    for r in chk[chk.warnings != ""].itertuples():
        w(f"  {r.node_id}: {r.warnings}")
    for r in link[link.link_status != "CANDIDATE"].itertuples():
        w(f"  {r.node_id}: 도로 노드 {r.dist_to_road_node_m} m ({r.link_status})")
    w(f"  통과 {int(nodes.j2_pass.sum())}개 / 실패 {int((~nodes.j2_pass).sum())}개: {nodes.loc[~nodes.j2_pass, 'node_id'].tolist()}")
    w("\n3. 역할 (t2_2026_rules.sensor_roles, J1·J2 적용)")
    w(f"  타깃 {len(roles['targets'])}개, pool {len(roles['pool'])}개, H {roles['H']}, 제외 {roles['dropped']}")
    w("\n4. 도로거리")
    q = pairs.road_dist_m.replace(np.inf, np.nan)
    w(f"  쌍 {len(pairs):,}개, 도달 불가 {int(np.isinf(pairs.road_dist_m).sum())}개, 중앙값 {q.median():,.0f} m, "
      f"우회비 > 2 {int((pairs.detour_ratio > 2).sum())}개")
    nn = pd.concat([pairs.groupby("node_a").road_dist_m.min(), pairs.groupby("node_b").road_dist_m.min()]).groupby(level=0).min()
    w(f"  최근접 이웃 도로거리 중앙값 {nn.median():,.0f} m (2025 18개: 708 m)")
    w("\n5. k=5 이웃 관측률 진단 (모델 실행 전, 규칙은 바꾸지 않음)")
    w(f"  k=5 이웃 블록 관측률: 타깃 평균 {avail.k5_block_obs_rate.mean():.2f} (최소 {avail.k5_block_obs_rate.min():.2f})")
    w(f"  블록당 관측된 k=5 이웃 수: 평균 {avail.k5_observed_per_block.mean():.2f} / IDW 후보 관측 수 평균 "
      f"{avail.idw_observed_per_block.mean():.1f}")
    w(f"  k=5 이웃이 하나도 관측되지 않은 블록 비율: 타깃 평균 {avail.k5_none_observed_share.mean():.3f}, "
      f"최대 {avail.k5_none_observed_share.max():.3f}")
    w(avail.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return "\n".join(L) + "\n"


def main() -> None:
    sel = pd.read_csv(OUT_SELECTION)
    pool = sel[sel.in_pool]
    geo = run_geocoding(geocoding_table(pool, pd.read_csv(GEO_2025, dtype={"district_code": str})))
    geo.to_csv(OUT_GEO, index=False, encoding="utf-8-sig")

    chk, _ = check_coordinates(geo)
    ok = chk[chk.valid & ~chk.outside_bbox & ~chk.address_not_gangnam]
    G = road_graph()
    nodes_df = ox.graph_to_gdfs(G, edges=False).reset_index()[["osmid", "y", "x"]]
    link = snap_to_nodes(ok, nodes_df)
    link["dist_to_road_edge_m"] = nearest_edge_dist(G, link)

    nodes = chk[["node_id", "lat", "lon", "location_basis", "geocoding_method", "warnings"]].merge(
        link[["node_id", "candidate_road_node", "dist_to_road_node_m", "dist_to_road_edge_m", "link_status"]],
        on="node_id", how="left").rename(columns={"candidate_road_node": "road_node"})
    nodes["j2_pass"] = nodes["link_status"].eq("CANDIDATE")
    z = np.load(TS_NPZ)
    roles = sensor_roles(sel, jump_excluded(z["values"], list(z["node_ids"])),
                         nodes.loc[~nodes.j2_pass, "node_id"].tolist() + sorted(set(pool.node_id) - set(nodes.node_id)))
    nodes["role"] = np.where(nodes.node_id.isin(roles["targets"]), "target",
                             np.where(nodes.node_id.isin(roles["pool"]), "neighbor", "dropped"))
    nodes["in_H"] = nodes.node_id.isin(roles["H"])
    pairs = pairwise_road_distance(G, nodes[nodes.j2_pass])
    avail = neighbor_availability(TS_NPZ, nodes, pairs[pairs.node_a.isin(roles["pool"]) & pairs.node_b.isin(roles["pool"])], roles)

    nodes.to_csv(OUT_NODES, index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
    report = build_report(geo, chk, link, nodes, pairs, roles, avail)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
