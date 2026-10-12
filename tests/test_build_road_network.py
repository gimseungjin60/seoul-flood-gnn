"""실행: python -m tests.test_build_road_network  (pytest로도 실행 가능). 네트워크 호출 없음."""
import networkx as nx
import pandas as pd

from src.data.build_road_network import build_report, haversine_to_many, snap_to_nodes, study_bbox

ROAD = pd.DataFrame({"osmid": [1, 2, 3], "y": [37.5000, 37.5100, 37.5200], "x": [127.0500, 127.0500, 127.0500]})


def stations() -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": ["A", "B", "C", "D"],
        "lat": [37.5002, 37.5101, 37.5000, 37.5300],   # D: 노드 3에서 약 1.1 km
        "lon": [127.0500, 127.0501, 127.0502, 127.0500],
        "location_basis": ["ADDRESS", "FACILITY_PROXY", "ADDRESS", "ADDRESS"],
    })


def test_bbox_and_distance():
    w, s, e, n = study_bbox(pd.Series([37.50, 37.52]), pd.Series([127.05, 127.07]), buffer_m=500)
    assert w < 127.05 and e > 127.07 and s < 37.50 and n > 37.52
    assert abs(haversine_to_many(37.5, 127.05, ROAD["y"].to_numpy(), ROAD["x"].to_numpy())[1] - 1112) < 5


def test_snap_rules():
    link = snap_to_nodes(stations(), ROAD, max_m=100).set_index("node_id")
    assert link.loc["A", "nearest_road_node"] == 1 and link.loc["A", "link_status"] == "CANDIDATE"
    assert link.loc["B", "is_facility_proxy"] and link.loc["B", "needs_review"]          # 시설 대용 표시
    assert link.loc["D", "link_status"] == "HOLD_FAR" and pd.isna(link.loc["D", "nearest_road_node"])  # 자동 연결 안 함
    assert link.loc["D", "candidate_road_node"] == 3                                        # 후보만 기록
    assert link.loc["A", "shares_road_node_with"] == "C" and link.loc["C", "needs_review"]  # 같은 노드 공유
    assert len(link) == 4                                                                  # 입력 지점만, 좌표 생성 없음


def test_report_mentions_caveats():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, length=100.0)
    G.add_edge(2, 3, length=100.0)
    link = snap_to_nodes(stations(), ROAD, max_m=100)
    link["dist_to_road_edge_m"] = [5.0, 5.0, 5.0, 60.0]
    edges = pd.DataFrame({"length": [100.0, 100.0], "highway": ["residential", "primary"]})
    excluded = pd.DataFrame({"node_id": ["X"], "geocoding_status": ["NOT_FOUND"]})
    text = build_report(G, (127.0, 37.4, 127.1, 37.6), ROAD, edges, link, excluded)
    assert "실제 하수관로 연결 관계와 다르다" in text
    assert "HOLD_FAR 1" in text and "X: 좌표 없음(NOT_FOUND)" in text
    assert "노드 3 / 도로 구간(edge) 2" in text


if __name__ == "__main__":
    test_bbox_and_distance()
    test_snap_rules()
    test_report_mentions_caveats()
    print("ok")
