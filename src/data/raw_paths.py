"""data/raw 원본 파일 경로를 한 곳에서 정의한다 (git 제외 폴더, 구조는 data/README.md 참고).

data/raw/
├── sewer/2025/tv_swm_wal_mea_2025{06,07,08}.csv      수위 월별 원본(서울 전체, 1분)
├── sewer/2026/하수관로_수위_현황_202609.csv           수위 2026-09 월별 원본
├── sewer/snapshot/서울시 하수관로 수위 현황.csv        최신 스냅샷(위치정보 컬럼 포함, 이력 아님)
├── rain/2025/서울시_강우량_정보_2025년{05..08}월.csv   강우량 월별(10분, cp949)
├── rain/2026/rain_2026{06,07,08}.csv                  강우량 2026-06~08 (포털 월별 다운로드)
└── rain/gauges/*.xlsx                                 강우량계 위치·관측지점 정보
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"

SEWER_2025 = RAW / "sewer" / "2025"
SEWER_2026 = RAW / "sewer" / "2026"
SEWER_SNAPSHOT = RAW / "sewer" / "snapshot"
RAIN_2025 = RAW / "rain" / "2025"
RAIN_2026 = RAW / "rain" / "2026"
RAIN_GAUGES = RAW / "rain" / "gauges"

SEWER_2025_CSVS = [SEWER_2025 / f"tv_swm_wal_mea_2025{m}.csv" for m in ("06", "07", "08")]
SEWER_202609_CSV = SEWER_2026 / "하수관로_수위_현황_202609.csv"
SEWER_SNAPSHOT_CSV = SEWER_SNAPSHOT / "서울시 하수관로 수위 현황.csv"
RAIN_2025_CSVS = {m: RAIN_2025 / f"서울시_강우량_정보_2025년{m}월.csv" for m in ("05", "06", "07", "08")}
RAIN_GAUGE_LOCATION_XLSX = RAIN_GAUGES / "서울시 강우량계 위치정보.xlsx"
