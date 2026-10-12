"""실행: python -m tests.test_build_station_graph  (pytest로도 실행 가능). 네트워크 호출 없음."""
import networkx as nx
import numpy as np
import pandas as pd

from src.data.build_station_graph import (annotate_edges, build_report, check_ids, graph_stats, knn_edges,
                                          pairwise_road_distance, radius_edges)

# 직선 도로 1-2-3-4 (각 100 m) + 5는 떨어진 도로 노드(연결 없음 → 도달 불가)
ROAD = nx.MultiDiGraph()
for a, b in [(1, 2), (2, 3), (3, 4)]:
    ROAD.add_edge(a, b, length=100.0)  # 한 방향만 → 무방향 변환 필요
ROAD.add_node(5)
STATIONS = pd.DataFrame({"node_id": ["S1", "S2", "S3", "S4"], "road_node": [1, 2, 3, 4],
                         "lat": [37.500, 37.5009, 37.5018, 37.5027], "lon": [127.0] * 4})
NODES = STATIONS.assign(is_facility_proxy=[False, True, False, False],
                        needs_review=[False, True, False, True])


def test_pairwise_is_undirected_road_distance():
    p = pairwise_road_distance(ROAD, STATIONS).set_index(["node_a", "node_b"])
    assert p.loc[("S1", "S4"), "road_dist_m"] == 300.0     # 역방향 간선만 있어도 무방향으로 계산
    assert len(p) == 6 and (p["straight_dist_m"] > 0).all()
    unreachable = pairwise_road_distance(ROAD, pd.concat([STATIONS, pd.DataFrame(
        {"node_id": ["S5"], "road_node": [5], "lat": [37.6], "lon": [127.0]})]))
    assert np.isinf(unreachable["road_dist_m"]).sum() == 4


def test_knn_union_no_self_loops():
    p = pairwise_road_distance(ROAD, STATIONS)
    e = knn_edges(p, 1)
    pairs = set(zip(e.node_a, e.node_b))
    assert not (e.node_a == e.node_b).any()
    # S2 → S1(동률 S1/S3 중 node_id 순), S3 → S2(동률 S2/S4 중 node_id 순) → 합집합 3간선
    assert pairs == {("S1", "S2"), ("S2", "S3"), ("S3", "S4")}
    assert e.set_index(["node_a", "node_b"]).loc[("S1", "S2"), "mutual"]


def test_radius_keeps_isolated_and_flags():
    p = pairwise_road_distance(ROAD, STATIONS)
    e = annotate_edges(radius_edges(p, 150), NODES)
    s = graph_stats(STATIONS["node_id"].tolist(), e)
    assert s["edges"] == 3 and s["isolated"] == [] and s["components"] == 1
    e0 = annotate_edges(radius_edges(p, 50), NODES)
    s0 = graph_stats(STATIONS["node_id"].tolist(), e0)
    assert s0["edges"] == 0 and len(s0["isolated"]) == 4 and s0["components"] == 4  # 억지 연결 없음
    assert e.set_index(["node_a", "node_b"]).loc[("S1", "S2"), "involves_facility_proxy"]
    assert (e["edge_type"] == "approx_road_proximity").all()


def test_check_ids_and_report():
    idc = check_ids(["S1", "S2", "S9"], pd.DataFrame({"node_id": ["S1", "S2", "S3", "S4"]}), STATIONS)
    assert idc["targets_not_in_water_level"] == ["S9"] and idc["targets_not_in_graph"] == ["S9"]
    assert idc["graph_not_in_targets"] == ["S3", "S4"]
    p = pairwise_road_distance(ROAD, STATIONS)
    graphs = {"k1": annotate_edges(knn_edges(p, 1), NODES)}
    stats = {"k1": graph_stats(STATIONS["node_id"].tolist(), graphs["k1"])}
    nodes = NODES.assign(idx=range(4), edge_far=False, coverage_ratio=0.99)
    text = build_report(nodes, pd.DataFrame({"node_id": ["S9"], "reason": "좌표 없음"}), p, graphs, stats, idc)
    assert "실제 하수관로 연결이 아니다" in text and "S9: 좌표 없음" in text


if __name__ == "__main__":
    test_pairwise_is_undirected_road_distance()
    test_knn_union_no_self_loops()
    test_radius_keeps_isolated_and_flags()
    test_check_ids_and_report()
    print("ok")
