"""원본 수위 CSV → 측정 지점(노드) 테이블 + 자치구 코드 1차 검증 + 품질 리포트.

실행: python -m src.data.clean_nodes
출력: data/processed/sewer_nodes_cleaned.csv, data/processed/data_quality_report.txt

202609 원본 컬럼(확인됨): 고유번호, 구분코드, 구분명, 측정일자, 측정수위, 통신상태
- 지점명 컬럼이 없음 → raw_point_name / cleaned_point_name 은 만들지 않는다.
- 좌표 컬럼이 없음 → latitude / longitude 는 만들지 않는다.
"""
import re
from pathlib import Path

import pandas as pd

from src.data.inspect_water_level import PROJECT_ROOT, RAW_CSV, detect_encoding, read_chunks

ID_COL, CODE_COL, NAME_COL, TIME_COL = "고유번호", "구분코드", "구분명", "측정일자"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_CSV = PROCESSED / "sewer_nodes_cleaned.csv"
OUT_REPORT = PROCESSED / "data_quality_report.txt"

# 검증용 참고표: 서울시 자치구 관례 순서(01 종로 ~ 25 강동).
# 확인 필요: 열린데이터광장 공식 코드 정의서와 대조하지 않았음. 불일치 시 수정하지 않고 검토 표시만 한다.
SEOUL_GU_REF = dict(enumerate(
    "종로 중 용산 성동 광진 동대문 중랑 성북 강북 도봉 노원 은평 서대문 마포 양천 "
    "강서 구로 금천 영등포 동작 관악 서초 강남 송파 강동".split(), start=1))
SEOUL_GU_REF = {f"{k:02d}": v for k, v in SEOUL_GU_REF.items()}

_INVISIBLE = re.compile(r"[​‌‍﻿]")
_WS = re.compile(r"\s+")  # 줄바꿈, 탭, NBSP(\xa0), 전각공백(　) 포함


def normalize_text(value) -> str | None:
    """공백/제어문자만 정리. 의미가 바뀌는 수정(단어 삭제, 치환)은 하지 않는다."""
    if value is None or pd.isna(value):
        return None
    s = _WS.sub(" ", _INVISIBLE.sub("", str(value))).strip()
    return s or None


def aggregate(path: Path = RAW_CSV) -> pd.DataFrame:
    """(고유번호, 구분코드, 구분명) 조합별 측정 건수와 기간. 원본 값 그대로 집계."""
    keys = [ID_COL, CODE_COL, NAME_COL]
    parts = [
        c.groupby(keys, dropna=False)[TIME_COL].agg(["size", "min", "max"])
        for c in read_chunks(path, usecols=keys + [TIME_COL])
    ]
    return (pd.concat(parts).groupby(level=keys, dropna=False)
            .agg({"size": "sum", "min": "min", "max": "max"})
            .reset_index()
            .rename(columns={ID_COL: "raw_node_id", CODE_COL: "raw_district_code",
                             NAME_COL: "raw_district_name", "size": "n_records",
                             "min": "first_measured", "max": "last_measured"}))


def build_nodes(combos: pd.DataFrame) -> pd.DataFrame:
    """조합 테이블 → 지점당 1행 + 검토 사유."""
    df = combos.assign(
        node_id=combos["raw_node_id"].map(normalize_text),
        district_code=combos["raw_district_code"].map(normalize_text),
        district_name=combos["raw_district_name"].map(normalize_text),
    )
    # 구분코드 ↔ 구분명 1:1 여부 (데이터 전체 기준)
    names_per_code = df.groupby("district_code", dropna=False)["district_name"].nunique()
    codes_per_name = df.groupby("district_name", dropna=False)["district_code"].nunique()

    join = lambda s: "|".join(sorted({x for x in s if x is not None}))
    nodes = df.groupby("node_id", dropna=False).agg(
        raw_node_id=("raw_node_id", join),
        district_code=("district_code", join),
        district_name=("district_name", join),
        n_records=("n_records", "sum"),
        first_measured=("first_measured", "min"),
        last_measured=("last_measured", "max"),
        n_raw_combos=("raw_node_id", "size"),
    ).reset_index()
    # 지점마다 측정 주기가 다름(1분/1시간/2시간 등) → 건수 대신 '측정 기간이 전체 기간의 몇 %인지'로 판단
    first = pd.to_datetime(nodes["first_measured"])
    last = pd.to_datetime(nodes["last_measured"])
    span = last - first
    nodes["coverage_pct"] = (span / (last.max() - first.min()) * 100).round(1)
    nodes["approx_interval_min"] = (span.dt.total_seconds() / 60 / (nodes["n_records"] - 1).clip(lower=1)).round(1)

    def reasons(r) -> str:
        out = []
        if pd.isna(r.node_id):
            out.append("고유번호 결측")
        if r.raw_node_id != r.node_id:
            out.append("고유번호 공백/제어문자 정리됨")
        if r.n_raw_combos > 1:
            out.append("동일 지점에 여러 자치구 코드/명 또는 원본 표기 혼재")
        if not r.district_code:
            out.append("구분코드 결측")
        if not r.district_name:
            out.append("구분명 결측")
        if r.district_code and r.district_name and "|" not in r.district_code:
            if names_per_code.get(r.district_code, 0) > 1:
                out.append("구분코드가 여러 구분명에 대응")
            if codes_per_name.get(r.district_name, 0) > 1:
                out.append("구분명이 여러 구분코드에 대응")
            ref = SEOUL_GU_REF.get(r.district_code)
            if ref is None:
                out.append("참고표에 없는 구분코드")
            elif ref != r.district_name:
                out.append(f"참고표 불일치(참고표: {ref})")
        # 고유번호 형식이 '구분코드-일련번호'로 보임(확인 필요) → 접두부가 다르면 검토
        if not pd.isna(r.node_id) and r.district_code and r.node_id.split("-")[0] != r.district_code:
            out.append("고유번호 접두부 ≠ 구분코드")
        # ponytail: 50% 기준은 임의값. 시계열 결측 분석(Week 3~) 후 조정
        if r.coverage_pct < 50:
            out.append(f"측정 기간 짧음(전체의 {r.coverage_pct}%, {r.n_records}건)")
        return "; ".join(out)

    nodes["review_reason"] = nodes.apply(reasons, axis=1)
    nodes["needs_review"] = nodes["review_reason"] != ""
    cols = ["node_id", "raw_node_id", "district_code", "district_name",
            "n_records", "first_measured", "last_measured", "coverage_pct", "approx_interval_min",
            "needs_review", "review_reason"]
    return nodes[cols].sort_values("node_id", ignore_index=True)


def write_report(nodes: pd.DataFrame, combos: pd.DataFrame, path: Path = RAW_CSV) -> str:
    total_rows = int(combos["n_records"].sum())
    n_cols = len(pd.read_csv(path, encoding=detect_encoding(path), nrows=0).columns)
    changed = combos[combos["raw_node_id"] != combos["raw_node_id"].map(normalize_text)]
    review = nodes[nodes["needs_review"]]
    rec = nodes["n_records"]
    L = [
        "서울시 하수관로 수위 현황 - 데이터 품질 리포트 (Week 1)",
        "=" * 60,
        f"원본 파일명: {path.name}",
        f"데이터 행 수: {total_rows:,}",
        f"컬럼 수: {n_cols}",
        f"실제 측정 지점 수(고유번호 기준): {len(nodes):,}",
        f"중복 지점 수(같은 고유번호에 원본 표기/자치구가 2개 이상): {nodes['review_reason'].str.contains('혼재').sum()}",
        "지점명 결측 수: 해당 없음 (원본에 지점명 컬럼이 없음)",
        f"고유번호 결측 수: {nodes['node_id'].isna().sum()}",
        f"자치구 정보 결측 수: 구분코드 {(nodes['district_code'] == '').sum()}, 구분명 {(nodes['district_name'] == '').sum()}",
        f"측정 기간: {nodes['first_measured'].min()} ~ {nodes['last_measured'].max()}",
        "",
        "[자치구별 지점 수]",
        nodes.groupby(["district_code", "district_name"]).size().to_string(),
        "",
        "[지점별 측정 건수 분포]",
        rec.describe().round(1).to_string(),
        "",
        "[추정 측정 주기(분) 분포 - 지점마다 다름, 공식 주기 확인 필요]",
        nodes["approx_interval_min"].round().value_counts().head(10).to_string(),
        "",
        "[측정 건수 하위 10개 지점 - 장기 결측/신규 설치 여부 확인 필요]",
        nodes.nsmallest(10, "n_records")[["node_id", "district_name", "n_records",
                                          "first_measured", "last_measured"]].to_string(index=False),
        "",
        "[정제 전/후 비교 예시 (고유번호)]",
        (changed[["raw_node_id"]].assign(node_id=changed["raw_node_id"].map(normalize_text))
         .head(10).to_string(index=False) if len(changed) else "변경된 값 없음 (공백/제어문자 문제 없음)"),
        "",
        f"[검토가 필요한 지점: {len(review)}개]",
        review[["node_id", "district_code", "district_name", "review_reason"]].to_string(index=False)
        if len(review) else "없음",
        "",
        "[현재 데이터에서 확인할 수 없는 정보]",
        "- 지점명(설치 위치 설명): 원본에 없음 → Week 2 Kakao 검색어로 쓸 정보 별도 확보 필요",
        "- 위도/경도 좌표: 원본에 없음",
        "- 측정수위 단위(m로 추정, 확인 필요) 및 지점별 관로 높이/위험 수위 기준",
        "- 구분코드 체계의 공식 정의(01~25 자치구 순서는 참고표 기준 추정, 확인 필요)",
        "- 고유번호 형식('구분코드-일련번호') 공식 정의 (확인 필요)",
        "- 통신상태: 202609 파일에는 '통신양호' 한 값만 존재 → 통신 불량 시 행 자체가 빠지는지 확인 필요",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"집계 중: {RAW_CSV.name}")
    combos = aggregate()
    nodes = build_nodes(combos)
    nodes.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # Excel 한글 깨짐 방지
    OUT_REPORT.write_text(write_report(nodes, combos), encoding="utf-8")
    print(f"지점 {len(nodes)}개 (검토 필요 {nodes['needs_review'].sum()}개)")
    print(f"→ {OUT_CSV}\n→ {OUT_REPORT}")


if __name__ == "__main__":
    main()
