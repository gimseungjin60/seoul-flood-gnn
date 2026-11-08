# 용도별 그림 모음

발표·실습일지에 실제로 넣은 그림을 용도별로 모은 복사본이다. 원본은 `outputs/figures/`에 있고 스크립트가 그곳에 다시 만든다. **원본을 다시 만들면 이 폴더는 자동으로 바뀌지 않는다.** 같은 그림을 다른 곳에 또 쓰면 여기에도 복사해 둔다.

## 중간발표 (`중간발표/`)
덱 내용: [docs/presentation/midterm_presentation.md](../presentation/midterm_presentation.md)

| 파일 | 원본 | 만드는 스크립트 |
|---|---|---|
| 슬라이드04_2025_센서18개_지도 | `slide/s04_sensor_map.png` | `src/analysis/story_figures.py` |
| 슬라이드09_2025_일강우량_이벤트 | `slide/s07_rain_events.png` | `src/analysis/slide_figures.py` |
| 슬라이드11_10분뒤_예측오차 | `slide/s10_before_after.png` | `story_figures.py` |
| 슬라이드13_모델별_호우구간_오차 | `slide/s09_model_error.png` | `slide_figures.py` |
| 슬라이드14_GNN_포함_모델별_오차 | `slide/s18_gnn_ci.png` | `story_figures.py` |
| 슬라이드15_거리_유사도_산점도 | `slide/s11_distance_similarity.png` | `slide_figures.py` |
| 백업3_1차진단_이웃평균_IDW | `slide/b03_loo_ratio.png` | `slide_figures.py` |
| 백업4_센서별_개선율 | `slide/b04_sensor_gain.png` | `slide_figures.py` |
| 백업5_사례_0014_0021_지도 / 시계열 | `slide/b05_case_map.png`, `b05_case_series.png` | `slide_figures.py` |

- 슬라이드 3의 강남역 침수 사진은 외부 사진이라 저장소에 없다. 백업 6(민감도)은 표라서 그림 파일이 없다.
- `slide/s10_idw_diff.png`는 최종 덱에 쓰지 않았다.

## 실습일지
| 폴더 | 그림 | 원본 |
|---|---|---|
| `6주차_실습일지/` | 그림1 수위 데이터 품질 분석, 그림2 학습 대상 20개 지점 위치정보 | 편집기 화면 캡처(6주차 HWP에서 꺼냄, 원본 파일 없음) |
| `7주차_실습일지/` | 그림1 공간 그래프 ~ 그림6 결측 현황 | `fig1_spatial_graph_k3`, `fig2_water_levels_full`, `fig2_water_levels_zoom`, `fig3_baseline_mae`, `fig3_baseline_rmse`, `fig4_observation_availability` (`src/visualization/make_figures.py`) |
| `9주차_실습일지/` | 그림1 2026 센서 배치, 그림2 2026 모델별 오차 | `slide/s2026_sensor_map.png`, `slide/s2026_gnn_ci.png` (`src/analysis/figures_2026.py`) |

일지 문서(HWP·docx)는 `docs/logs/`에 있다.
