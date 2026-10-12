"""관측 지점 간 도로망 최단거리 기반 공간 그래프 초안 (연구용 근사 그래프, GNN 학습 없음).

실행: python -m src.data.build_station_graph
입력(읽기만): data/processed/gangnam_station_road_link.csv, gangnam_road_network.graphml,
             gangnam_geocoding_input.csv(대상 20개), district_quality_nodes.csv(2025 수위 데이터 노드 ID·커버리지)
출력: data/processed/gangnam_station_pairwise_road_distance.csv, gangnam_station_graph_nodes.csv,
      gangnam_station_graph_{k2,k3,k4,radius500}.csv, gangnam_station_graph_report.txt, gangnam_station_graph_map.html

- 간선 = 도로망 최단거리상 가까운 지점 쌍(공간적 근사 관계). 실제 하수관로 연결·흐름 방향이 아니다(무방향).
- 도로망은 무방향으로 변환해 거리 계산(일방통행은 하수 흐름과 무관). 거리 = 매칭된 도로 노드 사이 최단경로 길이.
- 좌표 없는 지점은 그래프에서 제외하고 보고서에 기록한다. 좌표를 만들지 않는다.
"""
import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src.data.build_road_network import OUT as ROAD_OUT
from src.data.build_road_network import OUT_LINK, haversine_to_many
from src.data.clean_nodes import PROCESSED
from src.data.geocoder import OUT_CSV as RESULT_CSV
from src.data.prepare_geocoding_input import OUT_CSV as TARGET_CSV

QUALITY_NODES = PROCESSED / "district_quality_nodes.csv"
PAIRWISE_CSV = PROCESSED / "gangnam_station_pairwise_road_distance.csv"
NODES_CSV = PROCESSED / "gangnam_station_graph_nodes.csv"
REPORT = PROCESSED / "gangnam_station_graph_report.txt"
MAP_HTML = PROCESSED / "gangnam_station_graph_map.html"
K_VALUES, DEFAULT_K, RADIUS_M = (2, 3, 4), 3, 500
# 사전 고정 기준(결과를 보고 바꾸지 않음):
LONG_EDGE_IQR = 1.5   # 그래프 내 간선거리 > Q3 + 1.5·IQR → 긴 간선
DETOUR_MAX = 2.0      # 도로거리 / 직선거리 > 2.0 → 우회가 큰 간선(매칭·도로망 단절 의심)
EDGE_TYPE = "approx_road_proximity"


def pairwise_road_distance(G_road, stations: pd.DataFrame) -> pd.DataFrame:
    """stations(node_id, road_node, lat, lon) 모든 쌍의 무방향 도로망 최단거리(m)와 직선거리(m)."""
    U = G_road.to_undirected() if G_road.is_directed() else G_road
    road_nodes = dict(zip(stations["node_id"], stations["road_node"]))
    rows = []
    ids = list(stations["node_id"])
    for i, a in enumerate(ids):
        lengths = nx.single_source_dijkstra_path_length(U, road_nodes[a], weight="length")
        ra = stations.iloc[i]
        straight = haversine_to_many(ra["lat"], ra["lon"], stations["lat"].to_numpy(), stations["lon"].to_numpy())
        for j, b in enumerate(ids):
            if j <= i:
                continue
            d = lengths.get(road_nodes[b], np.inf)
            rows.append({"node_a": a, "node_b": b, "road_dist_m": round(float(d), 1),
                         "straight_dist_m": round(float(straight[j]), 1)})
    p = pd.DataFrame(rows)
    p["detour_ratio"] = (p["road_dist_m"] / p["straight_dist_m"]).round(3)
    return p


def _symmetric(pairs: pd.DataFrame) -> pd.DataFrame:
    rev = pairs.rename(columns={"node_a": "node_b", "node_b": "node_a"})
    return pd.concat([pairs, rev], ignore_index=True)


def knn_edges(pairs: pd.DataFrame, k: int) -> pd.DataFrame:
    """각 지점의 도로거리 k-최근접 이웃 → 무방향 합집합. 동률은 node_id 순으로 고정. 자기 루프 없음."""
    s = _symmetric(pairs)
    s = s[np.isfinite(s["road_dist_m"]) & (s["node_a"] != s["node_b"])]
    picks = (s.sort_values(["node_a", "road_dist_m", "node_b"]).groupby("node_a").head(k))
    picks = picks.assign(u=picks[["node_a", "node_b"]].min(axis=1), v=picks[["node_a", "node_b"]].max(axis=1))
    sel = picks.groupby(["u", "v"])["node_a"].agg(lambda x: "|".join(sorted(x))).rename("selected_by")
    e = sel.reset_index().rename(columns={"u": "node_a", "v": "node_b"})
    e = e.merge(pairs, on=["node_a", "node_b"], how="left")
    e["mutual"] = e["selected_by"].str.contains(r"\|")
    return e


def radius_edges(pairs: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    e = pairs[pairs["road_dist_m"] <= radius_m].copy()
    e["selected_by"] = f"road_dist<={radius_m:g}m"
    e["mutual"] = True
    return e


def annotate_edges(e: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    e = e.copy()
    proxy = set(nodes.loc[nodes["is_facility_proxy"], "node_id"])
    review = set(nodes.loc[nodes["needs_review"], "node_id"])
    e["edge_type"] = EDGE_TYPE
    e["involves_facility_proxy"] = e["node_a"].isin(proxy) | e["node_b"].isin(proxy)
    e["involves_review_node"] = e["node_a"].isin(review) | e["node_b"].isin(review)
    if len(e):
        q1, q3 = e["road_dist_m"].quantile([0.25, 0.75])
        e["long_edge"] = e["road_dist_m"] > q3 + LONG_EDGE_IQR * (q3 - q1)
    else:
        e["long_edge"] = False
    e["large_detour"] = e["detour_ratio"] > DETOUR_MAX
    cols = ["node_a", "node_b", "road_dist_m", "straight_dist_m", "detour_ratio", "selected_by", "mutual",
            "edge_type", "involves_facility_proxy", "involves_review_node", "long_edge", "large_detour"]
    return e[cols].sort_values(["node_a", "node_b"]).reset_index(drop=True)


def graph_stats(node_ids: list[str], e: pd.DataFrame) -> dict:
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(zip(e["node_a"], e["node_b"]))
    deg = pd.Series(dict(G.degree()))
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "isolated": sorted(nx.isolates(G)), "components": nx.number_connected_components(G),
            "component_sizes": sorted((len(c) for c in nx.connected_components(G)), reverse=True),
            "deg_min": int(deg.min()), "deg_median": float(deg.median()), "deg_max": int(deg.max()),
            "deg_counts": deg.value_counts().sort_index().to_dict(),
            "dist_min": e["road_dist_m"].min() if len(e) else None,
            "dist_median": e["road_dist_m"].median() if len(e) else None,
            "dist_max": e["road_dist_m"].max() if len(e) else None,
            "n_long": int(e["long_edge"].sum()) if "long_edge" in e else 0,
            "n_detour": int(e["large_detour"].sum()) if "large_detour" in e else 0}


def check_ids(targets: list[str], quality: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """대상 20개 ↔ 2025 수위 데이터 ID ↔ 그래프 노드 ID 정확 일치 검사(공백·형식 포함)."""
    wl = set(quality["node_id"])
    graph_ids = set(nodes["node_id"])
    return {
        "targets_not_in_water_level": sorted(set(targets) - wl),
        "graph_not_in_water_level": sorted(graph_ids - wl),
        "graph_not_in_targets": sorted(graph_ids - set(targets)),
        "targets_not_in_graph": sorted(set(targets) - graph_ids),
        "bad_format": sorted(i for i in set(targets) | graph_ids
                             if not isinstance(i, str) or i != i.strip() or len(i) != 7),
    }


def build_map(G_road, nodes: pd.DataFrame, e: pd.DataFrame, excluded: list[str]):
    U = G_road.to_undirected()
    rn = dict(zip(nodes["node_id"], nodes["road_node"]))
    m = folium.Map(tiles="OpenStreetMap")
    for r in e.itertuples():
        path = nx.shortest_path(U, rn[r.node_a], rn[r.node_b], weight="length")
        coords = [[U.nodes[n]["y"], U.nodes[n]["x"]] for n in path]
        c = "#dc2626" if (r.long_edge or r.large_detour) else "#f59e0b" if r.involves_review_node else "#2563eb"
        folium.PolyLine(coords, color=c, weight=4 if r.mutual else 2, opacity=0.8,
                        tooltip=f"{r.node_a} — {r.node_b}: 도로 {r.road_dist_m:,.0f} m "
                                f"(직선 {r.straight_dist_m:,.0f} m, 선택: {r.selected_by})").add_to(m)
    for r in nodes.itertuples():
        c = "#f59e0b" if r.is_facility_proxy else "#7c3aed" if r.needs_review else "#2563eb"
        flag = "시설 위치 대용" if r.is_facility_proxy else "도로 구간 거리 큼(검토)" if r.needs_review else "주소 기반"
        folium.CircleMarker([r.lat, r.lon], radius=8, color="#111", weight=1.5, fill=True, fill_color=c,
                            fill_opacity=0.95, tooltip=f"{r.node_id} ({flag})").add_to(m)
    m.fit_bounds([[nodes.lat.min(), nodes.lon.min()], [nodes.lat.max(), nodes.lon.max()]], padding=(30, 30))
    legend = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 12px;'
              'border:1px solid #999;border-radius:6px;font:13px sans-serif;color:#111;max-width:340px">'
              f'<b>관측 지점 그래프 (도로거리 k={DEFAULT_K}-NN, 무방향) — 연구용 근사</b>'
              '<div>선 = 도로망 최단경로, 굵은 선 = 상호 k-NN</div>'
              '<div><span style="color:#2563eb">━</span> 일반 간선 &nbsp;<span style="color:#f59e0b">━</span> 검토 노드 포함'
              ' &nbsp;<span style="color:#dc2626">━</span> 긴 간선/우회 큼</div>'
              '<div><span style="color:#2563eb">●</span> 주소 기반 <span style="color:#f59e0b">●</span> 시설 대용 '
              '<span style="color:#7c3aed">●</span> 도로 구간 거리 큼</div>'
              f'<div>제외(좌표 없음): {", ".join(excluded) or "없음"}</div>'
              '<div style="color:#b91c1c;margin-top:4px">간선 ≠ 실제 하수관로 연결, 흐름 방향 없음</div></div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m


def build_report(nodes, excluded_rows, pairs, graphs: dict, stats: dict, idc: dict) -> str:
    L = []
    w = L.append
    w("[강남] 관측 지점 공간 그래프 초안 리포트 (연구용 근사 그래프)")
    w("=" * 66)
    w("※ 간선은 도로망 최단거리상의 공간적 근접 관계이며 실제 하수관로 연결이 아니다.")
    w("   하수 흐름 방향·상하류 관계는 추정하지 않았다(모든 그래프 무방향). GNN 학습은 하지 않았다.")
    w("\n사용 기준 (결과 확인 전 고정)")
    w("  - 거리: 매칭된 도로 노드 사이 무방향 도로망 최단경로 길이(m). 지점→도로 노드 접근거리는 미포함")
    w(f"  - k-NN: k={K_VALUES}, 기본 k={DEFAULT_K}, 한쪽이라도 이웃으로 선택하면 간선 유지(합집합), 자기 루프 제외,"
      " 동률은 node_id 순")
    w(f"  - 반경 그래프: 도로거리 ≤ {RADIUS_M} m, 고립 노드는 억지로 연결하지 않음")
    w(f"  - 긴 간선: 해당 그래프 간선거리 > Q3 + {LONG_EDGE_IQR}·IQR / 큰 우회: 도로/직선 > {DETOUR_MAX}")

    w("\n1. ID 일치 검사 (대상 20개 ↔ 2025 수위 데이터 ↔ 그래프 노드)")
    for k, v in idc.items():
        w(f"  {k}: {v or '없음'}")
    w("  그래프 제외 지점(좌표 없음, 좌표 생성 안 함):")
    for r in excluded_rows.itertuples():
        w(f"    {r.node_id}: {r.reason}")

    w(f"\n2. 그래프 노드 {len(nodes)}개 (표시)")
    for r in nodes.itertuples():
        flags = [f for f, on in (("시설 위치 대용", r.is_facility_proxy),
                                 ("도로 구간 거리 큼", r.edge_far), ("검토 필요", r.needs_review)) if on]
        w(f"  [{r.idx:>2}] {r.node_id} 도로노드 {r.road_node} 커버리지 {r.coverage_ratio:.1%} {', '.join(flags)}")

    w(f"\n3. 지점 쌍 도로거리 ({len(pairs)}쌍)")
    w(f"  도로거리: 최소 {pairs.road_dist_m.min():,.0f} m, 중앙값 {pairs.road_dist_m.median():,.0f} m, "
      f"최대 {pairs.road_dist_m.max():,.0f} m / 도달 불가 쌍 {int((~np.isfinite(pairs.road_dist_m)).sum())}")
    w(f"  우회비(도로/직선): 중앙값 {pairs.detour_ratio.median():.2f}, 최대 {pairs.detour_ratio.max():.2f}")
    nn = _symmetric(pairs).groupby("node_a")["road_dist_m"].min()
    w(f"  지점별 최근접 이웃 도로거리: 최소 {nn.min():,.0f} m, 중앙값 {nn.median():,.0f} m, 최대 {nn.max():,.0f} m "
      f"({nn.idxmax()})")

    w("\n4. 그래프 비교")
    w(f"  {'그래프':<10} {'노드':>4} {'간선':>4} {'고립':>4} {'요소':>4} {'차수(min/med/max)':>18} "
      f"{'간선거리 m(min/med/max)':>26} {'긴간선':>5} {'큰우회':>5}")
    for name, s in stats.items():
        dd = (f"{s['dist_min']:,.0f}/{s['dist_median']:,.0f}/{s['dist_max']:,.0f}" if s["edges"] else "-")
        w(f"  {name:<10} {s['nodes']:>4} {s['edges']:>4} {len(s['isolated']):>4} {s['components']:>4} "
          f"{s['deg_min']:>6}/{s['deg_median']:g}/{s['deg_max']:<6} {dd:>26} {s['n_long']:>5} {s['n_detour']:>5}")
    for name, s in stats.items():
        w(f"  {name}: 연결요소 크기 {s['component_sizes']}, 고립 {s['isolated'] or '없음'}, 차수 분포 {s['deg_counts']}")

    w("\n5. 간선 검토 (긴 간선 / 큰 우회 / 검토 노드 포함)")
    for name, e in graphs.items():
        flagged = e[e["long_edge"] | e["large_detour"]]
        w(f"  [{name}] 긴 간선·큰 우회 {len(flagged)}개, 시설대용 포함 {int(e.involves_facility_proxy.sum())}개, "
          f"검토노드 포함 {int(e.involves_review_node.sum())}개")
        for r in flagged.itertuples():
            why = ", ".join(x for x, on in (("긴 간선", r.long_edge), ("큰 우회", r.large_detour)) if on)
            w(f"    {r.node_a} — {r.node_b}: 도로 {r.road_dist_m:,.0f} m / 직선 {r.straight_dist_m:,.0f} m "
              f"(×{r.detour_ratio:.2f}) {why}")
    return "\n".join(L) + "\n"


def main() -> None:
    link = pd.read_csv(OUT_LINK)
    targets = pd.read_csv(TARGET_CSV)["node_id"].tolist()
    quality = pd.read_csv(QUALITY_NODES)
    G_road = ox.load_graphml(ROAD_OUT["network.graphml"])

    usable = link[link["link_status"] == "CANDIDATE"].copy()
    nodes = pd.DataFrame({
        "node_id": usable["node_id"], "lat": usable["station_lat"], "lon": usable["station_lon"],
        "road_node": usable["candidate_road_node"].astype(int),
        "location_basis": usable["location_basis"], "is_facility_proxy": usable["is_facility_proxy"].astype(bool),
        "dist_to_road_node_m": usable["dist_to_road_node_m"], "dist_to_road_edge_m": usable["dist_to_road_edge_m"],
        "edge_far": usable["dist_to_road_edge_m"] > 50, "needs_review": usable["needs_review"].astype(bool),
    }).sort_values("node_id").reset_index(drop=True)
    nodes = nodes.merge(quality[["node_id", "coverage_ratio"]], on="node_id", how="left")
    nodes.insert(0, "idx", range(len(nodes)))
    status = pd.read_csv(RESULT_CSV).set_index("node_id")["geocoding_status"]
    excluded = [{"node_id": t, "reason": f"좌표 없음 — 지오코딩 상태 {status.get(t, '결과 파일에 없음')}"}
                for t in targets if t not in set(nodes["node_id"])]
    excluded_rows = pd.DataFrame(excluded)

    pairs = pairwise_road_distance(G_road, nodes)
    graphs = {f"k{k}": annotate_edges(knn_edges(pairs, k), nodes) for k in K_VALUES}
    graphs[f"radius{RADIUS_M}"] = annotate_edges(radius_edges(pairs, RADIUS_M), nodes)
    stats = {n: graph_stats(nodes["node_id"].tolist(), e) for n, e in graphs.items()}
    idc = check_ids(targets, quality, nodes)

    pairs.to_csv(PAIRWISE_CSV, index=False, encoding="utf-8-sig")
    nodes.to_csv(NODES_CSV, index=False, encoding="utf-8-sig")
    for n, e in graphs.items():
        e.to_csv(PROCESSED / f"gangnam_station_graph_{n}.csv", index=False, encoding="utf-8-sig")
    build_map(G_road, nodes, graphs[f"k{DEFAULT_K}"], excluded_rows["node_id"].tolist()).save(str(MAP_HTML))
    report = build_report(nodes, excluded_rows, pairs, graphs, stats, idc)
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
