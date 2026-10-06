"""원본 하수관로 수위 CSV 구조 검사 (원본은 읽기만 한다).

실행: python -m src.data.inspect_water_level [csv경로]
파일이 ~1.2GB / 2,200만 행이라 chunksize 단위로 읽어 통계를 누적한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "하수관로_수위_현황_202609.csv"
CHUNKSIZE = 1_000_000


def detect_encoding(path: Path) -> str:
    """UTF-8로 앞부분 디코딩이 되면 utf-8-sig, 아니면 cp949 (서울 열린데이터광장 기본)."""
    with path.open("rb") as f:
        head = f.read(1_000_000)
    try:
        head.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError as e:
        if e.start > len(head) - 4:  # 청크 경계에서 멀티바이트 문자가 잘린 경우
            return "utf-8-sig"
        return "cp949"


def clean_header(name: str) -> str:
    # 202609 파일 헤더 첫 글자가 BOM 대신 '?'(0x3F)로 저장되어 있음 → 제거
    return name.strip().lstrip("﻿?").strip()


def read_chunks(path: Path = RAW_CSV, usecols=None, chunksize: int = CHUNKSIZE):
    """모든 값을 문자열로 읽는다(앞자리 0 보존: '06' → 6 변환 방지)."""
    enc = detect_encoding(path)
    header = pd.read_csv(path, encoding=enc, nrows=0).columns
    names = [clean_header(c) for c in header]
    return pd.read_csv(
        path, encoding=enc, dtype=str, keep_default_na=False, na_values=[""],
        header=0, names=names, usecols=usecols, chunksize=chunksize,
    )


def guess_type(values: pd.Series) -> str:
    v = values.dropna()
    if v.empty:
        return "empty"
    if pd.to_numeric(v, errors="coerce").notna().all():
        return "numeric" if not v.str.match(r"^0\d").any() else "code(leading zero)"
    if pd.to_datetime(v, errors="coerce", format="mixed").notna().all():
        return "datetime"
    return "text"


def main(path: Path = RAW_CSV) -> None:
    print(f"파일: {path}")
    if not path.exists():
        print("파일 없음")
        sys.exit(1)
    print(f"파일 크기: {path.stat().st_size / 1024**2:,.1f} MB")
    print(f"인코딩 추정: {detect_encoding(path)}")

    n_rows, nulls, uniques, row_hashes, head, types = 0, None, {}, [], None, None
    for chunk in read_chunks(path):
        if head is None:
            head = chunk.head()
            types = {c: guess_type(chunk[c].head(10_000)) for c in chunk.columns}
            nulls = pd.Series(0, index=chunk.columns)
            uniques = {c: set() for c in chunk.columns}
        n_rows += len(chunk)
        nulls += chunk.isna().sum()
        for c in chunk.columns:
            uniques[c].update(chunk[c].unique())
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        print(f"  ...{n_rows:,}행 읽음", end="\r")

    n_unique_rows = len(np.unique(np.concatenate(row_hashes)))
    n_unique = {c: len(s) for c, s in uniques.items()}

    print(f"\n행 개수: {n_rows:,}")
    print(f"컬럼 목록: {list(head.columns)}")
    print("\n[컬럼별 타입(상위 1만 행 기준 추정) / 결측 / 고유값]")
    for c in head.columns:
        print(f"  {c:<12} {types[c]:<20} 결측={nulls[c]:,}  고유값={n_unique[c]:,}")
    print("\n[상위 5개 행]")
    print(head.to_string())
    print(f"\n완전 중복 행 개수: {n_rows - n_unique_rows:,}")

    # 지점 식별 후보: 텍스트/코드형이면서 고유값이 전체 행보다 훨씬 적고(반복 측정) 1보다 큰 컬럼
    print("\n[지점 식별 후보 컬럼]")
    for c in head.columns:
        if types[c] in ("text", "code(leading zero)") and 1 < n_unique[c] < n_rows * 0.01:
            sample = sorted(x for x in uniques[c] if isinstance(x, str))[:10]
            print(f"  {c}: 고유값 {n_unique[c]:,}개, 예시 {sample}")
    small = {c: sorted(map(str, uniques[c])) for c in head.columns if n_unique[c] <= 30}
    print("\n[고유값 30개 이하 컬럼의 전체 값]")
    for c, vals in small.items():
        print(f"  {c}: {vals}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_CSV)
