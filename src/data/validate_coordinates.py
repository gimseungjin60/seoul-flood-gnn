"""강남 지오코딩 결과 좌표 품질 검증 + Folium 지도 (API 호출 없음, 좌표 수정/생성 없음).

실행: python -m src.data.validate_coordinates
입력: data/processed/gangnam_geocoding_result.csv
출력: data/processed/gangnam_geocoding_map.html, gangnam_coordinate_validation_report.txt

- 지도에는 위도/경도가 숫자이고 지구 좌표 범위 안인 노드만 표시. 좌표 없는 노드는 마커 없음.
- 검사 결과는 '추가 확인 필요' 신호일 뿐 오류 판정이 아니다(가까운 좌표 ≠ 잘못된 좌표).
"""
import html
from itertools import combinations

import folium
import numpy as np
import pandas as pd

from src.data.clean_nodes import PROCESSED
from src.data.geocoder import OUT_CSV as RESULT_CSV

MAP_HTML = PROCESSED / "gangnam_geocoding_map.html"
REPORT = PROCESSED / "gangnam_coordinate_validation_report.txt"
# ponytail: 강남구 대략 외곽 사각형(경계 폴리곤 아님, 약간 여유). 정밀 판정은 행정경계 폴리곤으로 교체.
GANGNAM_BBOX = {"lat": (37.45, 37.54), "lon": (127.01, 127.13)}
DUP_M, NEAR_M = 1.0, 100.0  # 동일 좌표 / 근접 좌표 기준(m)
COLORS = {"ADDRESS": "#2563eb", "FACILITY_PROXY": "#f59e0b", "REVIEW": "#dc2626"}
LABELS = {"ADDRESS": "주소 기반 좌표", "FACILITY_PROXY": "시설 위치 대용 좌표 (검토 필요)",
          "REVIEW": "검증 경고 있음 (근접·이상치·범위 밖 등)"}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6_371_000 * 2 * np.arcsin(np.sqrt(a)))


def check_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """노드별 형식/범위/연구범위/근접/이상치 검사 → (노드 표, 근접 쌍 표). 좌표는 바꾸지 않는다."""
    d = df.copy()
    lat, lon = pd.to_numeric(d["latitude"], errors="coerce"), pd.to_numeric(d["longitude"], errors="coerce")
    d["lat"], d["lon"] = lat, lon
    d["missing"] = d["latitude"].isna() & d["longitude"].isna()
    d["non_numeric"] = (d["latitude"].notna() & lat.isna()) | (d["longitude"].notna() & lon.isna()) \
        | (lat.isna() ^ lon.isna())
    d["out_of_range"] = lat.notna() & lon.notna() & ~(lat.between(-90, 90) & lon.between(-180, 180))
    d["valid"] = lat.notna() & lon.notna() & ~d["out_of_range"]
    d["outside_bbox"] = d["valid"] & ~(lat.between(*GANGNAM_BBOX["lat"]) & lon.between(*GANGNAM_BBOX["lon"]))
    d["address_not_gangnam"] = d["valid"] & ~d["matched_address"].fillna("").str.contains("강남구")

    v = d[d["valid"]]
    pairs = [(a.node_id, b.node_id, haversine_m(a.lat, a.lon, b.lat, b.lon))
             for a, b in combinations(v.itertuples(), 2)]
    pairs = pd.DataFrame(pairs, columns=["node_a", "node_b", "dist_m"]).sort_values("dist_m")
    nn = {n: pairs.loc[(pairs.node_a == n) | (pairs.node_b == n), "dist_m"].min() for n in v["node_id"]}
    d["nearest_m"] = d["node_id"].map(nn)
    d["duplicate"] = d["node_id"].isin(pd.concat([pairs.loc[pairs.dist_m < DUP_M, c] for c in ("node_a", "node_b")]))
    d["near"] = d["node_id"].isin(pd.concat([pairs.loc[pairs.dist_m < NEAR_M, c] for c in ("node_a", "node_b")]))
    # 이상치: 최근접 이웃 거리가 Q3 + 1.5·IQR 초과(다른 노드들과 동떨어진 점)
    q1, q3 = d["nearest_m"].quantile([0.25, 0.75])
    d["isolated"] = d["nearest_m"] > q3 + 1.5 * (q3 - q1)
    flags = ["non_numeric", "out_of_range", "outside_bbox", "address_not_gangnam", "duplicate", "near", "isolated"]
    d["warnings"] = d[flags].apply(lambda r: ", ".join(f for f in flags if r[f]), axis=1)
    d["marker_class"] = np.where(d["warnings"] != "", "REVIEW",
                                 np.where(d["location_basis"] == "FACILITY_PROXY", "FACILITY_PROXY", "ADDRESS"))
    return d, pairs


def _popup(r) -> str:
    e = lambda x: html.escape(str(x)) if pd.notna(x) else "-"
    return (f"<b>{e(r.node_id)}</b><br>원본 위치정보: {e(r.search_keyword)}<br>검색된 주소: {e(r.matched_address)}"
            + (f"<br>매칭 시설: {e(r.matched_place)}" if pd.notna(r.matched_place) else "")
            + f"<br>위도: {r.lat:.6f}<br>경도: {r.lon:.6f}<br>좌표 출처: {e(r.location_basis)}"
            f"<br>검토 필요: {e(r.needs_review)}<br>검증 경고: {e(r.warnings or '없음')}")


def build_map(d: pd.DataFrame) -> folium.Map:
    v = d[d["valid"]]
    m = folium.Map(tiles="OpenStreetMap")
    for r in v.itertuples():
        c = COLORS[r.marker_class]
        folium.CircleMarker([r.lat, r.lon], radius=8, color="#111" if r.needs_review else c, weight=2,
                            fill=True, fill_color=c, fill_opacity=0.85, tooltip=r.node_id,
                            popup=folium.Popup(_popup(r), max_width=360)).add_to(m)
    if len(v):
        m.fit_bounds([[v.lat.min(), v.lon.min()], [v.lat.max(), v.lon.max()]], padding=(30, 30))
    rows = "".join(f'<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                   f'background:{COLORS[k]};margin-right:6px"></span>{LABELS[k]}</div>' for k in COLORS)
    legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 12px;'
              f'border:1px solid #999;border-radius:6px;font:13px sans-serif;color:#111">'
              f'<b>강남 수위 관측 지점 ({len(v)}개 표시)</b>{rows}'
              f'<div style="margin-top:4px">검은 테두리 = needs_review=True</div>'
              f'<div>좌표 없음(마커 없음): {", ".join(d.loc[~d["valid"], "node_id"]) or "없음"}</div></div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m


def marker_ids(m: folium.Map) -> list[str]:
    """지도에 실제로 추가된 마커의 노드 ID(툴팁) 목록."""
    return [t.text for c in m._children.values() if isinstance(c, folium.CircleMarker)
            for t in c._children.values() if isinstance(t, folium.Tooltip)]


def build_report(d: pd.DataFrame, pairs: pd.DataFrame, n_markers: int) -> str:
    L = []
    w = L.append
    v = d[d["valid"]]
    w("[강남] 지오코딩 좌표 품질 검증 리포트")
    w("=" * 60)
    w(f"입력: {RESULT_CSV.relative_to(PROCESSED.parents[1])}")
    w(f"\n전체 대상 지점: {len(d)}")
    w(f"좌표 확보 지점: {len(v)}")
    w(f"지도에 표시된 마커: {n_markers} (유효 좌표 수와 일치: {n_markers == len(v)})")
    w(f"주소 기반 좌표: {int((v['location_basis'] == 'ADDRESS').sum())}")
    w(f"시설 위치 대용 좌표: {int((v['location_basis'] == 'FACILITY_PROXY').sum())} "
      f"({', '.join(v.loc[v['location_basis'] == 'FACILITY_PROXY', 'node_id'])})")
    w("좌표 미확정 지점 (마커 없음):")
    for r in d[~d["valid"]].itertuples():
        w(f"  {r.node_id} [{r.geocoding_status}] {r.search_keyword}")

    w("\n1. 좌표 형식 및 범위 검사")
    w(f"  숫자 변환 불가: {int(d['non_numeric'].sum())} / 위도·경도 범위 밖: {int(d['out_of_range'].sum())} / "
      f"결측(좌표 미확정): {int(d['missing'].sum())}")
    w(f"  강남구 대략 범위(위도 {GANGNAM_BBOX['lat']}, 경도 {GANGNAM_BBOX['lon']}, 경계 폴리곤 아님) 밖: "
      f"{', '.join(d.loc[d['outside_bbox'], 'node_id']) or '없음'}")
    w(f"  검색 주소에 '강남구' 없음: {', '.join(d.loc[d['address_not_gangnam'], 'node_id']) or '없음'}")
    w(f"  유효 좌표 범위: 위도 {v.lat.min():.5f} ~ {v.lat.max():.5f}, 경도 {v.lon.min():.5f} ~ {v.lon.max():.5f}")

    w(f"\n2. 중복·근접 좌표 검사 (동일 < {DUP_M:g} m, 근접 < {NEAR_M:g} m, 하버사인 거리)")
    close = pairs[pairs["dist_m"] < NEAR_M]
    w(f"  동일 좌표 쌍: {int((pairs['dist_m'] < DUP_M).sum())} / 근접 쌍: {len(close)}")
    for r in close.itertuples():
        w(f"    {r.node_a} ↔ {r.node_b}: {r.dist_m:,.0f} m")
    w("  가장 가까운 5쌍:")
    for r in pairs.head(5).itertuples():
        w(f"    {r.node_a} ↔ {r.node_b}: {r.dist_m:,.0f} m")
    w(f"  최근접 이웃 거리: 최소 {v.nearest_m.min():,.0f} m, 중앙값 {v.nearest_m.median():,.0f} m, "
      f"최대 {v.nearest_m.max():,.0f} m")

    w("\n3. 분포 이상치 (최근접 이웃 거리 > Q3 + 1.5·IQR)")
    iso = d[d["isolated"]]
    w("  " + (", ".join(f"{r.node_id}({r.nearest_m:,.0f} m)" for r in iso.itertuples()) or "없음"))

    w("\n4. 추가 검토가 필요한 지점과 사유")
    for r in d.itertuples():
        reasons = []
        if not r.valid:
            reasons.append(f"좌표 없음({r.geocoding_status}) — 지도 확인 또는 현장 정보로 위치 확정 필요")
        if r.location_basis == "FACILITY_PROXY":
            reasons.append(f"시설 '{r.matched_place}' 위치를 대신 사용 — 원문은 시설 앞 보도/도로라 맨홀과 수십 m 차이 가능")
        if r.near:
            reasons.append(f"다른 노드와 {r.nearest_m:,.0f} m 이내 — 서로 다른 맨홀일 수 있으나 같은 주소 기준점을 공유했는지 확인")
        if r.isolated:
            reasons.append(f"최근접 노드까지 {r.nearest_m:,.0f} m로 분포상 동떨어짐 — 오지오코딩 여부 확인(정상일 수 있음)")
        if r.outside_bbox or r.address_not_gangnam:
            reasons.append("강남구 범위/주소 불일치")
        if reasons:
            w(f"  {r.node_id}:")
            for x in reasons:
                w(f"    - {x}")
    w("\n주의")
    w("  - 강남구 안에 있다는 것은 맨홀 위치 확인이 아니다. 주소 기반 좌표도 건물 기준점이라 맨홀과 오차가 있다.")
    w("  - 근접/이상치 표시는 추가 확인 신호이며 오류 판정이 아니다. 좌표는 수정하지 않았다.")
    return "\n".join(L) + "\n"


def main() -> None:
    df = pd.read_csv(RESULT_CSV, dtype={"district_code": str})
    d, pairs = check_coordinates(df)
    m = build_map(d)
    markers = marker_ids(m)
    if len(markers) != int(d["valid"].sum()):
        raise SystemExit(f"마커 수 {len(markers)} ≠ 유효 좌표 수 {int(d['valid'].sum())}")
    m.save(str(MAP_HTML))
    report = build_report(d, pairs, len(markers))
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
