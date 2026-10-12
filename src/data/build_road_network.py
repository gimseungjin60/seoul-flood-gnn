"""OSMnx 강남 도로망(drive) 수집 + 관측 지점 → 최근접 도로망 노드 연결 후보 (GNN 학습 없음).

실행: python -m src.data.build_road_network
입력: data/processed/gangnam_geocoding_result.csv (읽기만 함)
출력: data/processed/gangnam_road_nodes.csv, gangnam_road_edges.csv, gangnam_road_network.graphml,
      gangnam_station_road_link.csv, gangnam_road_network_report.txt, gangnam_road_network_map.html

주의: 도로망 근접성은 실제 하수관로 연결 관계가 아니다. 여기서 만드는 것은 '연결 후보'(근사치)이며,
     관측 지점 간 연결은 확정하지 않는다. 좌표 없는 노드는 제외하고 좌표를 만들지 않는다.
"""
import html
import math

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

from src.data.clean_nodes import PROCESSED
from src.data.geocoder import OUT_CSV as RESULT_CSV
from src.data.inspect_water_level import PROJECT_ROOT
from src.data.validate_coordinates import check_coordinates

NETWORK_TYPE = "drive"
BUFFER_M = 500
NODE_MAX_M = 100  # 최근접 도로망 노드가 이보다 멀면 자동 연결 보류(HOLD_FAR)
EDGE_REVIEW_M = 50  # 도로 구간까지 이보다 멀면 연결 후보는 유지하되 검토 대상
OSM_CACHE = PROJECT_ROOT / "data" / "raw" / "osm_cache"  # data/raw는 git 제외
OUT = {k: PROCESSED / f"gangnam_road_{k}" for k in ("nodes.csv", "edges.csv", "network.graphml",
                                                     "network_report.txt", "network_map.html")}
OUT_LINK = PROCESSED / "gangnam_station_road_link.csv"
EARTH_M = 6_371_000


def study_bbox(lat: pd.Series, lon: pd.Series, buffer_m: float = BUFFER_M) -> tuple[float, float, float, float]:
    """관측 지점 경계 사각형 + buffer_m → (west, south, east, north)."""
    dlat = buffer_m / 111_320
    dlon = buffer_m / (111_320 * math.cos(math.radians(lat.mean())))
    return lon.min() - dlon, lat.min() - dlat, lon.max() + dlon, lat.max() + dlat


def haversine_to_many(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat), np.radians(lats)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2
    return EARTH_M * 2 * np.arcsin(np.sqrt(a))


def snap_to_nodes(stations: pd.DataFrame, road_nodes: pd.DataFrame, max_m: float = NODE_MAX_M) -> pd.DataFrame:
    """stations(node_id, lat, lon, location_basis) → 최근접 도로망 노드, 거리(m), 연결 후보 상태.
    road_nodes(osmid, y, x). 거리 > max_m 이면 HOLD_FAR (자동 연결 안 함)."""
    ys, xs, ids = road_nodes["y"].to_numpy(), road_nodes["x"].to_numpy(), road_nodes["osmid"].to_numpy()
    rows = []
    for r in stations.itertuples():
        dist = haversine_to_many(r.lat, r.lon, ys, xs)
        i = int(np.argmin(dist))
        far = dist[i] > max_m
        rows.append({"node_id": r.node_id, "station_lat": r.lat, "station_lon": r.lon,
                     "location_basis": r.location_basis,
                     "is_facility_proxy": r.location_basis == "FACILITY_PROXY",
                     "nearest_road_node": None if far else ids[i],
                     "candidate_road_node": ids[i], "road_node_lat": ys[i], "road_node_lon": xs[i],
                     "dist_to_road_node_m": round(float(dist[i]), 1),
                     "link_status": "HOLD_FAR" if far else "CANDIDATE"})
    out = pd.DataFrame(rows)
    shared = out["candidate_road_node"].duplicated(keep=False)
    out["shares_road_node_with"] = [
        ", ".join(out.loc[shared & (out["candidate_road_node"] == n) & (out["node_id"] != s), "node_id"])
        for s, n in zip(out["node_id"], out["candidate_road_node"])]
    out["needs_review"] = (out["link_status"] != "CANDIDATE") | out["is_facility_proxy"] | shared
    return out


def fetch_graph(bbox):
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OSM_CACHE)
    return ox.graph_from_bbox(bbox, network_type=NETWORK_TYPE)


def nearest_edge_dist(G, link: pd.DataFrame) -> pd.Series:
    """도로 구간(edge)까지 수직 거리(m) — 투영 좌표계에서 계산. 참고용."""
    Gp = ox.project_graph(G)
    geoms = [ox.projection.project_geometry(shapely.Point(lon, lat), crs="EPSG:4326", to_crs=Gp.graph["crs"])[0]
             for lat, lon in zip(link["station_lat"], link["station_lon"])]
    _, d = ox.distance.nearest_edges(Gp, [g.x for g in geoms], [g.y for g in geoms], return_dist=True)
    return pd.Series(np.round(d, 1), index=link.index)


def build_map(edges_gdf, link: pd.DataFrame, excluded: pd.DataFrame):
    m = folium.Map(tiles="OpenStreetMap")
    folium.GeoJson(edges_gdf[["geometry"]], name="도로망(drive)",
                   style_function=lambda _: {"color": "#6b7280", "weight": 1.5, "opacity": 0.7}).add_to(m)
    color = {"CANDIDATE": "#2563eb", "HOLD_FAR": "#dc2626"}
    for r in link.itertuples():
        c = "#f59e0b" if r.is_facility_proxy else color[r.link_status]
        folium.PolyLine([[r.station_lat, r.station_lon], [r.road_node_lat, r.road_node_lon]],
                        color=c, weight=2, dash_array="4 4").add_to(m)
        folium.CircleMarker([r.road_node_lat, r.road_node_lon], radius=4, color="#111", fill=True,
                            fill_color="#111", fill_opacity=1, tooltip=f"도로 노드 {r.candidate_road_node}").add_to(m)
        popup = (f"<b>{r.node_id}</b><br>좌표 출처: {r.location_basis}<br>최근접 도로 노드: {r.candidate_road_node}"
                 f"<br>노드까지 {r.dist_to_road_node_m:.1f} m / 도로 구간까지 {r.dist_to_road_edge_m:.1f} m"
                 f"<br>상태: {r.link_status}<br>검토 필요: {r.needs_review}"
                 + (f"<br>같은 도로 노드 공유: {html.escape(r.shares_road_node_with)}" if r.shares_road_node_with else ""))
        folium.CircleMarker([r.station_lat, r.station_lon], radius=8, color="#111" if r.needs_review else c,
                            weight=2, fill=True, fill_color=c, fill_opacity=0.9, tooltip=r.node_id,
                            popup=folium.Popup(popup, max_width=320)).add_to(m)
    m.fit_bounds([[link.station_lat.min(), link.station_lon.min()],
                  [link.station_lat.max(), link.station_lon.max()]], padding=(30, 30))
    legend = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 12px;'
              'border:1px solid #999;border-radius:6px;font:13px sans-serif;color:#111">'
              f'<b>관측 지점 {len(link)}개 ↔ 최근접 도로 노드 (연결 후보, 확정 아님)</b>'
              '<div><span style="color:#2563eb">●</span> 주소 기반, 노드 ≤ '
              f'{NODE_MAX_M} m (CANDIDATE)</div>'
              '<div><span style="color:#f59e0b">●</span> 시설 위치 대용 좌표</div>'
              f'<div><span style="color:#dc2626">●</span> 노드 &gt; {NODE_MAX_M} m (HOLD_FAR, 자동 연결 보류)</div>'
              '<div>● 검정 작은 점 = 도로망 노드, 점선 = 지점→노드, 회색 = 도로망</div>'
              '<div>검은 테두리 = 검토 필요</div>'
              f'<div>제외(좌표 없음): {", ".join(excluded["node_id"]) or "없음"}</div>'
              '<div style="margin-top:4px;color:#b91c1c">도로망 근접성 ≠ 실제 하수관로 연결</div></div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m


def build_report(G, bbox, nodes_df, edges_df, link, excluded) -> str:
    L = []
    w = L.append
    w("[강남] OSMnx 도로망 수집 및 관측 지점 연결 후보 리포트")
    w("=" * 64)
    w("※ 도로망의 공간적 근접성은 실제 하수관로 연결 관계와 다르다. 아래 연결은 연구용 근사치(후보)이며")
    w("   실제 하수관로 토폴로지를 확보한 것이 아니다. 관측 지점 간 연결은 확정하지 않았다.")
    w(f"\n입력: {RESULT_CSV.name} (읽기만 함, 변경 없음)")
    w(f"범위: 좌표 확보 {len(link)}개 지점 경계 사각형 + {BUFFER_M} m → "
      f"W {bbox[0]:.5f}, S {bbox[1]:.5f}, E {bbox[2]:.5f}, N {bbox[3]:.5f}")
    w(f"제외(좌표 없음, 좌표 생성 안 함): " + ", ".join(f"{r.node_id}[{r.geocoding_status}]" for r in excluded.itertuples()))
    w(f"\n1. 도로망 (OSMnx {ox.__version__}, network_type='{NETWORK_TYPE}', OpenStreetMap)")
    w(f"  노드 {len(nodes_df):,} / 도로 구간(edge) {len(edges_df):,} / 총 길이 {edges_df['length'].sum() / 1000:,.1f} km")
    wcc = list(nx.weakly_connected_components(G))
    largest = max(wcc, key=len)
    w(f"  약연결 요소 {len(wcc)}개, 최대 요소 노드 {len(largest):,}")
    hw = edges_df["highway"].value_counts().head(8)
    w("  도로 등급(상위): " + ", ".join(f"{k} {v}" for k, v in hw.items()))

    w(f"\n2. 관측 지점별 최근접 도로망 노드 (하버사인 거리, 자동 연결 기준 ≤ {NODE_MAX_M} m)")
    w(f"  {'node_id':<8} {'basis':<15} {'road_node':>12} {'노드(m)':>8} {'구간(m)':>8}  상태      비고")
    for r in link.itertuples():
        note = []
        if r.is_facility_proxy:
            note.append("시설 위치 대용")
        if r.shares_road_node_with:
            note.append(f"노드 공유: {r.shares_road_node_with}")
        if r.candidate_road_node not in largest:
            note.append("최대 연결요소 밖")
        w(f"  {r.node_id:<8} {r.location_basis:<15} {r.candidate_road_node:>12} {r.dist_to_road_node_m:>8.1f} "
          f"{r.dist_to_road_edge_m:>8.1f}  {r.link_status:<9} {'; '.join(note)}")
    d = link["dist_to_road_node_m"]
    w(f"  노드 거리: 최소 {d.min():.1f} m, 중앙값 {d.median():.1f} m, 최대 {d.max():.1f} m")
    w(f"  도로 구간 거리: 최소 {link['dist_to_road_edge_m'].min():.1f} m, "
      f"중앙값 {link['dist_to_road_edge_m'].median():.1f} m, 최대 {link['dist_to_road_edge_m'].max():.1f} m")

    w("\n3. 연결 보류 / 검토 대상")
    for r in link[link["needs_review"]].itertuples():
        why = []
        if r.link_status == "HOLD_FAR":
            why.append(f"최근접 도로 노드 {r.dist_to_road_node_m:.0f} m > {NODE_MAX_M} m → 자동 연결 보류")
        if r.is_facility_proxy:
            why.append("시설 위치 대용 좌표 → 맨홀 실제 위치와 오차 가능, 연결 노드가 달라질 수 있음")
        if r.shares_road_node_with:
            why.append(f"{r.shares_road_node_with}와 같은 도로 노드로 연결됨 → 그래프에서 구분 안 될 수 있음")
        if r.dist_to_road_edge_m > EDGE_REVIEW_M:
            why.append(f"도로(drive) 구간까지 {r.dist_to_road_edge_m:.0f} m > {EDGE_REVIEW_M} m → 단지·교내 도로 등 "
                       "drive 망에 없는 길 옆일 수 있음, 연결 노드 확인 필요")
        w(f"  {r.node_id}: " + " / ".join(why))
    for r in excluded.itertuples():
        w(f"  {r.node_id}: 좌표 없음({r.geocoding_status}) → 이번 분석 제외")
    c = link["link_status"].value_counts()
    w(f"\n요약: CANDIDATE {c.get('CANDIDATE', 0)} / HOLD_FAR {c.get('HOLD_FAR', 0)} / 제외 {len(excluded)} / "
      f"검토 필요 {int(link['needs_review'].sum())}")
    w("\n주의: 최근접 도로 노드는 대부분 교차로다. 주소 기반 좌표도 건물 기준점이라 실제 맨홀과 다른 노드로")
    w("      연결될 수 있다. 연결 후보는 현장/관망도 정보로 확인 전까지 확정이 아니다.")
    return "\n".join(L) + "\n"


def main() -> None:
    res = pd.read_csv(RESULT_CSV, dtype={"district_code": str})
    d, _ = check_coordinates(res)
    stations, excluded = d[d["valid"]], d[~d["valid"]]
    bbox = study_bbox(stations["lat"], stations["lon"])
    G = fetch_graph(bbox)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_df = nodes_gdf.reset_index()[["osmid", "y", "x", "street_count"]]
    edges_df = edges_gdf.reset_index()[["u", "v", "key", "length", "highway", "name", "oneway"]]
    for c in ("highway", "name"):
        edges_df[c] = edges_df[c].map(lambda x: "|".join(map(str, x)) if isinstance(x, list) else x)
    link = snap_to_nodes(stations, nodes_df)
    link["dist_to_road_edge_m"] = nearest_edge_dist(G, link)
    link["needs_review"] |= link["dist_to_road_edge_m"] > EDGE_REVIEW_M

    nodes_df.to_csv(OUT["nodes.csv"], index=False, encoding="utf-8-sig")
    edges_df.to_csv(OUT["edges.csv"], index=False, encoding="utf-8-sig")
    ox.save_graphml(G, OUT["network.graphml"])
    link.to_csv(OUT_LINK, index=False, encoding="utf-8-sig")
    build_map(edges_gdf, link, excluded).save(str(OUT["network_map.html"]))
    report = build_report(G, bbox, nodes_df, edges_df, link, excluded)
    OUT["network_report.txt"].write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
