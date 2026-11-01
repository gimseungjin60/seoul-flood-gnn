# data/ 폴더 안내

원본(`raw/`)은 용량이 커서(약 7 GB) git에 올리지 않는다. 경로는 `src/data/raw_paths.py` 한 곳에서 정의한다. 원본은 절대 수정하지 않고, 가공 결과는 `processed/`에만 저장한다.

## raw/ (git 제외)

| 경로 | 내용 | 기간 | 크기 | 비고 |
|---|---|---|---|---|
| `sewer/2025/tv_swm_wal_mea_2025{06,07,08}.csv` | 하수관로 수위 월별 원본(서울 전체, 1분, 위치정보 컬럼 포함) | 2025-06~08 | 1.7·2.0·2.2 GB | 강남구 센서 18개 분석의 입력 |
| `sewer/2026/하수관로_수위_현황_202609.csv` | 하수관로 수위 월별(서울 전체, 1분, 위치정보 없음, cp949) | 2026-09 | 1.2 GB | 지점 테이블(`sewer_nodes_cleaned.csv`)의 출처. 강남 111개 센서 |
| `sewer/2026/DrainpipeMonitoringInfo_2026{06,07,08,09}.csv` | 하수관로 수위 월별 현행화본(서울 전체, 위치정보 포함, cp949, 날짜 `01-JUN-26 00:00:00` 형식) | 2026-06~09 | 2.05·2.14·2.18·2.11 GB | 2026-11-01 서울시 현행화 안내 후 열린데이터광장 OA-2527에서 받음. 원본 zip과 응답 헤더는 `sewer/2026/zip/`. 강남 6~8월 센서 116개. 신규 센서 다수는 약 2시간 주기로 40~45분 기록, 75~87분 공백(초 단위 시각). 값 -1(오류 코드)과 07-07 13:49~14:45 등 시스템 이상 구간이 있음 → `gangnam2026_timeseries_quality_report.txt` |
| `sewer/snapshot/서울시 하수관로 수위 현황.csv` | 최신 스냅샷(위치정보 포함, cp949) | 2026-10-18 01:43~02:23 | 1.7 MB | 이력이 아니라 위치 마스터로만 사용 |
| `rain/2025/서울시_강우량_정보_2025년{05..08}월.csv` | 서울시 강우량계 10분 자료(cp949) | 2025-05~08 | 각 약 9 MB | |
| `rain/2026/rain_2026{06,07,08}.csv` | 서울시 강우량계 10분 자료(포털 월별 다운로드, cp949) | 2026-06~08 | 각 약 9 MB | `hdr_*.txt`는 다운로드 응답 헤더 기록 |
| `rain/gauges/서울시 강우량계 위치정보.xlsx` | 강우량계 48개소 위경도 | - | 15 KB | 센서별 최근접 배정에 사용 |
| `rain/gauges/서울시 강우량계 관측지점 정보_20251120123239.xlsx` | 기상청 관측지점 정보로 보이는 파일 | - | 16 KB | 사용하지 않음(`docs/06`) |
| `osm_cache/` | OSM 도로망 다운로드 캐시(자동 생성) | - | - | 없어도 됨 |

### 새 원본을 받을 때
1. 종류와 기간에 맞는 폴더에 넣는다(`sewer/<연도>/`, `rain/<연도>/`). 원본 파일 이름은 바꾸지 않는다.
2. `src/data/raw_paths.py`에 경로 상수를 추가한다.
3. 이 표에 한 줄을 추가한다(기간, 크기, 용도).
4. 다운로드한 파일은 신뢰할 수 없는 입력으로 본다. 스크립트는 raw 폴더 밖에 두고 `python -I`로 읽는다.

### 없는 자료
- **2026-09 강우**: 아직 받지 않음.

## processed/ (git 추적)

가공 결과만 둔다. 파일 이름은 `<대상>_<내용>.<확장자>`이고, 사람이 읽는 리포트는 `*_report.txt`다.

| 접두어 | 내용 |
|---|---|
| `data_quality_*`, `district_quality_*`, `sewer_nodes_cleaned.csv` | 서울 전체 지점 정제와 자치구별 품질 |
| `gangnam_202508_quality*`, `gangnam_location_match*` | 강남구 센서 선정과 위치 매칭 |
| `gangnam_geocoding_*`, `gangnam_coordinate_validation_report.txt` | 좌표 확보(지오코딩)와 검증 |
| `gangnam_road_*`, `gangnam_station_graph_*`, `gangnam_station_*` | 도로망, 센서 그래프, 센서별 도로 연결·거리·강우량계 배정 |
| `gangnam_station_timeseries*.npz`, `gangnam_timeseries_*` | 18개 센서 1분 시계열과 품질 |
| `gangnam_forecast_*`, `gangnam_baseline_*` | 시간 예측(T1) 데이터셋과 기준 모델 |
| `rainfall_*`, `gangnam_rain_*` | 강우 10분 격자, 이벤트, 강우 효과 실험 |
| `gangnam_spatial_signal_*`, `gangnam_t2_*` | 공간 신호 진단과 T2 센서 하나 가리고 맞히기(LOSO) 결과 |
| `gangnam2026_*` | 2026-06~08 강남 시계열(품질 규칙 R1·R2 적용), 센서 선정(core 20 / neighbor 34 / excluded 62), 품질 리포트. `python -m src.data.prepare_station_timeseries_2026` |
| `gangnam2026_geocoding_*`, `gangnam2026_station_*` | pool 54개 좌표(2025 재사용 17, Kakao 35, 실패 2), 도로거리, 역할(타깃 17 / pool 48), k=5 이웃 관측률. `python -m src.data.build_station_graph_2026` (도로망 graphml은 재생성 가능, git 제외) |

- 5 MB가 넘는 재생성 가능 결과와 예측 npz(`gangnam_t2_loso_predictions.npz`)는 `.gitignore`로 제외했다. 재생성 방법은 루트 `README.md` 12절이다.
