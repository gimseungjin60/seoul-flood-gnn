# seoul-flood-gnn

그래프 신경망(GNN) 기반 서울시 하수관로 침수 위험 예측 시스템

## 1. 프로젝트 소개

서울시 하수관로 수위 센서는 일부 지점에만 설치되어 있어 센서가 없는 곳의 수위와 침수 위험을 직접 알기 어렵다.
기존 센서의 수위 데이터와 주변 환경 데이터를 이용해 지점 간 공간 관계를 그래프로 구성하고,
GNN으로 센서가 없는 지점의 수위를 예측한 뒤 위험도를 지도에 표시한다.

| 팀원 | 담당 |
|---|---|
| 최진호 | 도로망/근접 그래프 구축, GNN 모델링(GCN/GraphSAGE), DEM, 강수량 결합, Masking 실험, MAE/RMSE 평가 |
| 김승진 | 원본 데이터 관리, 지점 데이터 정제/검증, Kakao 좌표 확보, Streamlit + Folium 대시보드, 위험도 시각화 |

## 2. 현재 개발 단계: Week 3 완료 (GNN 학습 전)

| 주차 | 내용 | 상태 |
|---|---|---|
| 1 | 원본 다운로드·관리, 지점 추출·정제, 자치구 코드 1차 검증 | 완료 |
| 2 | 위치정보 매칭·품질 검증, 대상 자치구(강남) 선정, 지오코딩 입력 준비 | 완료 |
| 3 | 좌표 확보(18/20)·검증, 도로망, 공간 그래프, 1분 시계열(18노드), 예측 데이터셋, 기준 모델, 그림 4종 | 완료 |
| 4~ | 공간 신호 진단, 강수량 결합, GCN/GraphSAGE, 마스킹 실험, 대시보드 | **미착수** |

- 대상: 서울 강남구 수위 센서 18개 × 2025-06-01~08-31 (1분 간격, 132,480스텝). 20개 대상 지점 중 23-0011(NOT_FOUND), 23-0018(MULTIPLE)은 좌표를 만들지 않고 제외했다. 23-0007, 23-0014는 시설 위치 대용 좌표다.
- 기준 모델(test, m): h=10분 MAE 0.0073(persistence) / 0.0072(Ridge AR), h=30분 MAE 0.0106 / 0.0102. **개선 폭이 작다.** 평균 오차는 평상시 값이 지배하므로 성능 근거로 쓰지 않는다.
- 강우 효과(선형 Ridge): 강우 특징을 더해도 평균 오차(MAE)는 개선되지 않고 극단(q99, RMSE)만 개선. 강우 제거 후에도 거리 감쇠는 약함 → [docs/07](docs/07_rain_effect_results.md).
- 강수량: 서울시 강우량계 10분 자료(2025-05~08) 정렬·검증 완료, 유의 강우 이벤트 10개(train 6 / val 2 / test 2). 모델 입력에는 아직 결합하지 않았다 → [docs/06](docs/06_rainfall_data.md).
- **아직 없는 것:** GNN 모델, 침수·위험 라벨(수위만 있음), 관로 제원/DEM, 대시보드.
- 그래프는 도로망 근접 기반 연구용 근사이며 실제 하수관로 연결·흐름 방향이 아니다.
- 실습일지 주차와의 대응: 일지 6주차(10/5~10/11) = 위 Week 2, 일지 7주차(10/12~10/18) = 위 Week 3.
- 상세 평가와 로드맵은 [docs/](docs/) 참고 (아래 13절).

## 3. 김승진 담당 업무

- 하수관로 원본 데이터 다운로드 및 관리
- 지점명 텍스트 정제
- 자치구 코드 기반 1차 검증
- Kakao Local API를 이용한 지점 좌표 확보
- 좌표 획득 실패 지점 수동 보정
- Streamlit + Folium 기반 대시보드, 위험도 판정 및 시각화
- 실제 수위 / 예측 수위 비교 화면, 경고 UI

## 4. 데이터 출처

서울 열린데이터광장 — "서울시 하수관로 수위 현황"

## 5. 원본 데이터 파일명

`data/raw/하수관로_수위_현황_202609.csv`

- 크기 약 1.1GB, 22,565,058행 → **GitHub에 올리지 않는다** (`.gitignore` 처리, GitHub 100MB 제한 초과).
  팀원은 열린데이터광장에서 같은 파일을 받아 `data/raw/`에 넣는다.
- 원본은 절대 수정하지 않는다. 가공 결과는 `data/processed/`에만 저장한다.

## 6. 폴더 구조

```
seoul-flood-gnn/
├── data/
│   ├── raw/                  # 원본 (git 제외)
│   └── processed/            # 가공 결과: 지점 테이블, 매칭·품질 리포트, 강남 좌표·도로망·그래프·시계열·데이터셋·기준 모델 결과
├── src/
│   ├── data/                 # 정제, 위치 매칭, 지오코딩, 도로망, 그래프, 시계열, 예측 데이터셋
│   ├── models/               # baselines.py (persistence, 노드별 Ridge AR). GNN은 아직 없음
│   └── visualization/        # make_figures.py
├── tests/                    # 모듈별 셀프 테스트 (test_*.py)
├── outputs/figures/          # 그림 4종 PNG + index.html
├── docs/                     # 진행 평가·로드맵·데이터셋 설계·발표 안 (md 문서는 여기서 관리)
├── notebooks/
├── .env.example  .gitignore  requirements.txt  main.py
```

## 7. 실행 방법 (Windows, VS Code 터미널, Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.inspect_water_level   # 원본 구조 검사 (약 2~3분)
python main.py                           # 지점 테이블 + 품질 리포트 생성 (약 30초)
python -m tests.test_clean_nodes         # 정제/검증 로직 셀프 테스트
```

지오코딩은 `.env.example`을 `.env`로 복사한 뒤 `KAKAO_REST_API_KEY`를 채워야 한다 (`.env`는 git에 올리지 않는다).

> 8~10절은 Week 1 기준(2026-09 원본, 전 구 690개 지점)이다. 강남 분석(2025-06~08)은 12절 참고.

콘솔에 한글이 깨지면 `$env:PYTHONIOENCODING="utf-8"` 후 실행.

## 8. 데이터 처리 흐름

```
raw CSV (cp949, 22.5M행)
  → 100만 행 단위 청크 읽기 (메모리 절약, 모든 값 문자열로 읽어 '06' 같은 앞자리 0 보존)
  → (고유번호, 구분코드, 구분명) 조합별 측정 건수 / 최초·최종 측정시각 집계
  → 텍스트 정제 (공백·줄바꿈·제어문자만 정리, 의미 변경 없음)
  → 자치구 1차 검증 → 검토 사유(review_reason) 기록
  → sewer_nodes_cleaned.csv (utf-8-sig, Excel에서 바로 열림) + data_quality_report.txt
```

## 9. 현재 확인된 데이터 구조 (202609)

| 컬럼 | 내용 | 결측 | 고유값 |
|---|---|---|---|
| 고유번호 | 지점 ID, 형식 `구분코드-일련번호` (예: `06-0002`) | 0 | **690** |
| 구분코드 | `01`~`25` 자치구 코드 | 0 | 25 |
| 구분명 | 자치구 약칭 (`동대문`, `중` = 중구, "구" 미포함) | 0 | 25 |
| 측정일자 | `2026-09-01 00:00:00.0` 형식, 2026-09-01 ~ 09-30 | 0 | 2,493,119 |
| 측정수위 | 숫자 | 0 | 265 |
| 통신상태 | `통신양호` 한 값만 존재 | 0 | 1 |

- 인코딩: **cp949**. 헤더 첫 글자가 BOM 대신 `?`로 저장되어 있어(`?고유번호`) 코드에서 제거한다.
- 완전 중복 행 0개.
- 측정 지점은 **690개**다. 기획서의 346개와 다르다 → 팀에서 확인 필요.
- 구분코드 ↔ 구분명은 1:1로 일치하고, 고유번호 접두부 = 구분코드가 690개 모두에서 성립한다.
- 지점마다 측정 주기가 다르다 (추정 주기 약 1분: 540개, 약 1시간: 98개, 약 2시간: 43개, 기타: 9개).
- 측정 기간이 짧은 지점 19개 (예: 21-0052~0055는 9/11에 1건만 존재) → `needs_review=True`.

`sewer_nodes_cleaned.csv` 컬럼:
`node_id, raw_node_id, district_code, district_name, n_records, first_measured, last_measured, coverage_pct, approx_interval_min, needs_review, review_reason`

## 10. 아직 확인되지 않은 사항

- **지점명 컬럼이 원본에 없다.** 그래서 `raw_point_name` / `cleaned_point_name`은 만들지 않았다. Week 2 Kakao 검색에 쓸 지점 위치 정보는 별도 데이터로 확보해야 한다.
- **좌표(위도/경도)가 원본에 없다.** 그래서 `latitude` / `longitude`는 만들지 않았다.
- 측정수위 단위 (m로 추정, 확인 필요)
- 구분코드 공식 정의: 01 종로 ~ 25 강동 관례 순서와 일치하지만 공식 코드 정의서와 대조하지 않음 (확인 필요)
- 고유번호 형식의 공식 정의 (확인 필요)
- 통신 불량일 때 행이 빠지는지, 다른 값으로 기록되는지 (확인 필요)
- 기획서 346개와 실제 690개의 차이 원인 (확인 필요)

## 11. 다음 단계

우선순위순 (자세한 근거와 판정 기준은 [docs/01_progress_and_roadmap.md](docs/01_progress_and_roadmap.md)):

1. ~~강수량 확보 및 시간축 정렬~~ 완료(서울시 강우량계, [docs/06](docs/06_rainfall_data.md)). 센서별 최근접 강우량계 배정도 완료
2. ~~공간 신호 진단~~ 완료 → [docs/05](docs/05_spatial_signal_result.md): 센서 간 공통 신호는 있으나 도로거리로는 설명되지 않음(Ridge 17개 입력이 상수 대비 0.50배, 거리 가중 IDW는 단순 평균과 비슷)
3. (선택) 비선형 baseline으로 강우 효과 재확인 → [docs/07](docs/07_rain_effect_results.md) 5절
4. 과제 분리 확정: T1 시간 예측 / T2 센서 제거(마스킹) 추정, 위험 라벨 정의
5. GCN/GraphSAGE 학습 (가장 강한 비-GNN baseline을 이기는지 검증)
6. 노드 마스킹 실험, 대시보드(Streamlit + Folium), 위험도 시각화

## 12. 강남구 분석 파이프라인 재현 순서 (2025-06~08)

원본 `data/raw/tv_swm_wal_mea_2025{06,07,08}.csv`가 필요하다(git 제외, 열린데이터광장에서 직접 다운로드).

```powershell
python -m src.data.analyze_location_match       # 시계열 고유번호 ↔ 위치정보 매칭 검증 (data/raw/서울시 하수관로 수위 현황.csv 필요)
python -m src.data.analyze_gangnam_quality      # 강남 2025-08 노드 품질, 6~8월 비교
python -m src.data.analyze_district_quality      # 자치구별 안정 노드 비교 → 강남 선정 근거
python -m src.data.prepare_geocoding_input       # 강남 HIGH 20개 지오코딩 입력
python -m src.data.geocoder                      # Kakao 지오코딩 (.env에 KAKAO_REST_API_KEY 필요)
python -m src.data.geocode_retry                 # 실패 지점 시설명 재검색
python -m src.data.validate_coordinates          # 좌표 검증 + 지도
python -m src.data.build_road_network            # OSMnx 도로망 (Overpass 호출)
python -m src.data.build_station_graph           # 도로거리 k-NN / 반경 그래프
python -m src.data.prepare_station_timeseries    # 18개 노드 1분 수위 행렬 + 마스크 (약 3분)
python -m src.data.build_forecast_dataset        # 10/30분 예측 샘플·시간순 분할·정규화 통계
python -m src.models.baselines                   # 기준 모델(마지막 관측값, 노드별 Ridge) 평가
python -m src.data.prepare_rainfall             # 강우량 10분 격자·품질·이벤트·수위 반응 (data/raw/서울시_강우량_정보_2025년{05..08}월.csv 필요)
python -m src.data.assign_rain_gauges          # 센서별 최근접 강우량계 배정 (data/raw/서울시 강우량계 위치정보.xlsx 필요)
python -m src.models.rain_baselines             # 강우 입력 효과(노드별 Ridge + 강우 특징) 비교
python -m src.analysis.spatial_signal           # 공간 신호 진단 (GNN 학습 전, test 미사용) + fig5/fig6
python -m src.analysis.spatial_signal_rain_adjusted  # 강우 제거 후 공간 진단 + fig7
python -m src.visualization.make_figures         # outputs/figures PNG + index.html
```

- 그래프 간선은 도로망 근접 기반 연구용 근사이며 실제 하수관로 연결이 아니다. GNN은 아직 학습하지 않았다.
- 용량 때문에 git에 올리지 않은 재생성 가능 파일(`.gitignore` 참고):
  `gangnam_station_timeseries.csv`, `gangnam_station_observation_mask.csv`(같은 내용이 커밋된 `gangnam_station_timeseries.npz`에 있음),
  `gangnam_forecast_samples_h{10,30}.csv`.
- **원본 CSV 없이 이어서 작업하는 방법:** clone 후 `pip install -r requirements.txt` → `python -m src.data.build_forecast_dataset`(npz에서 샘플 CSV 재생성, CSV가 없으면 npz만으로 검증) → `python -m src.models.baselines`.
  결과는 커밋된 기준 모델 지표와 1e-15 수준까지 일치함을 확인했다(2026-10-17, Python 3.13, numpy 2.5, pandas 3.0).
  원본(`data/raw`)은 시계열을 처음부터 다시 만들거나 강수량 정렬 등 새 분석을 할 때만 필요하다.
- `gangnam_road_network.graphml`은 OSM 데이터가 시간에 따라 바뀌므로 재현성을 위해 커밋한다.
- 테스트: `python -m tests.<테스트파일명>` (예: `python -m tests.test_baselines`)

## 13. 문서 (docs/)

md 문서는 `docs/`에서 관리한다.

| 파일 | 내용 |
|---|---|
| [01_progress_and_roadmap.md](docs/01_progress_and_roadmap.md) | 진행상황 평가, 문제 진단, 로드맵, 판정 기준 |
| [02_dataset_design.md](docs/02_dataset_design.md) | 데이터셋 설계 고려사항, 분할·평가 설계, 체크리스트 |
| [03_midterm_with_dataset.md](docs/03_midterm_with_dataset.md) | 중간발표 안 A (데이터셋 포함) |
| [04_midterm_without_dataset.md](docs/04_midterm_without_dataset.md) | 중간발표 안 B (데이터셋 제외) |
| [05_spatial_signal_result.md](docs/05_spatial_signal_result.md) | 공간 신호 진단 결과 (Phase A) |
| [06_rainfall_data.md](docs/06_rainfall_data.md) | 강수량 데이터 명세, 품질, 이벤트, 수위 반응 시차, 입력 규칙 (Phase B) |
| [07_rain_effect_results.md](docs/07_rain_effect_results.md) | 강우 효과: T1 Ridge+강우 비교(사전 규칙상 실패), 강우 제거 후 공간 진단 |
