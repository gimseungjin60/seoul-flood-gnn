"""수위 예측 데이터셋 설계: 결측 분석, 결측 처리 비교안, 샘플 인덱스, 시간순 분할, 정규화 통계 (학습 없음).

실행: python -m src.data.build_forecast_dataset
입력(읽기만): gangnam_station_timeseries.{csv,npz}, gangnam_station_observation_mask.csv,
             gangnam_station_graph_nodes.csv
출력: data/processed/gangnam_station_timeseries_locf5.npz  (처리값 + 처리코드, 원본과 별도)
      data/processed/gangnam_forecast_samples_h{10,30}.csv  (GNN·기준모델 공용 샘플 인덱스)
      data/processed/gangnam_forecast_norm_stats.csv        (학습 구간 관측값만으로 계산)
      data/processed/gangnam_forecast_dataset_report.txt

샘플 정의 (anchor t = 입력 창의 마지막 시각 인덱스):
  입력 = 시각 [t-W+1, t] (W=60), 정답 = 시각 t+h (h=10 또는 30), 노드 18개 전부 같은 t를 공유.
분할: 시간축 인덱스 기준 앞 70% / 다음 15% / 마지막 15%. 샘플은 입력 시작(t-W+1)과 정답(t+h)이 모두
  같은 분할 안에 있을 때만 그 분할에 속하고, 경계에 걸친 샘플은 버린다(purge).
"""
import numpy as np
import pandas as pd

from src.data.clean_nodes import PROCESSED
from src.data.prepare_station_timeseries import NODES_CSV, OUT_MASK, OUT_NPZ, OUT_VALUES, gap_runs

W, HORIZONS, LOCF_LIMIT = 60, (10, 30), 5
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
OUT_FILLED = PROCESSED / "gangnam_station_timeseries_locf5.npz"
OUT_SAMPLES = {h: PROCESSED / f"gangnam_forecast_samples_h{h}.csv" for h in HORIZONS}
OUT_NORM = PROCESSED / "gangnam_forecast_norm_stats.csv"
REPORT = PROCESSED / "gangnam_forecast_dataset_report.txt"
FILL_OBSERVED, FILL_LOCF, FILL_MISSING = 0, 1, 2


def load_and_verify():
    """npz와 CSV 두 형식을 모두 읽어 시간축·열 순서·마스크 일치를 재검증."""
    z = np.load(OUT_NPZ)
    values, mask = z["values"].astype(np.float64), z["mask"].astype(bool)
    times = pd.DatetimeIndex(z["times"])
    nodes = list(z["node_ids"])
    v_csv = pd.read_csv(OUT_VALUES, index_col=0, parse_dates=True)
    m_csv = pd.read_csv(OUT_MASK, index_col=0, parse_dates=True)
    order = pd.read_csv(NODES_CSV).sort_values("idx")["node_id"].tolist()
    checks = {
        "times_equal_csv": bool((v_csv.index == times).all() and (m_csv.index == times).all()),
        "times_1min_strict": bool((np.diff(times.to_numpy().astype("datetime64[s]").astype(np.int64)) == 60).all()),
        "range": f"{times[0]} ~ {times[-1]} ({len(times):,})",
        "columns_equal_graph_order": list(v_csv.columns) == order == list(m_csv.columns) == nodes,
        "values_equal_csv": bool(np.allclose(v_csv.to_numpy(), values, equal_nan=True, atol=1e-6)),
        "mask_equal_csv": bool((m_csv.to_numpy().astype(bool) == mask).all()),
        "mask_equals_notnan": bool((np.isfinite(values) == mask).all()),
    }
    return values, mask, times, nodes, checks


def locf_causal(values: np.ndarray, limit: int = LOCF_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """직전 관측값을 결측 시작 후 최대 limit분까지만 채움(과거 정보만). 원본은 수정하지 않음.
    반환: (채운 값 행렬, 처리코드 0=원본관측 / 1=직전값 채움 / 2=미처리 결측)."""
    filled = pd.DataFrame(values).ffill(limit=limit).to_numpy()
    code = np.full(values.shape, FILL_MISSING, dtype=np.int8)
    code[np.isfinite(values)] = FILL_OBSERVED
    code[~np.isfinite(values) & np.isfinite(filled)] = FILL_LOCF
    return filled, code


def gap_length_rule_fill_count(mask: np.ndarray, limit: int = LOCF_LIMIT) -> int:
    """(비교용) '전체 길이 ≤ limit인 결측 구간만 채움' 규칙이 채웠을 칸 수 — 구간 끝(미래)을 알아야 하는 규칙."""
    n = 0
    for j in range(mask.shape[1]):
        col = mask[:, j]
        runs = gap_runs(col)
        x = np.diff(np.concatenate([[1], col.astype(np.int8), [1]]))
        starts = np.where(x == -1)[0]
        n += int(sum(r for s, r in zip(starts, runs) if r <= limit and s > 0))
    return n


def split_bounds(n: int) -> dict[str, tuple[int, int]]:
    out, start = {}, 0
    for i, (name, frac) in enumerate(SPLITS):
        end = n if i == len(SPLITS) - 1 else start + int(round(n * frac))
        out[name] = (start, end)
        start = end
    return out


def window_counts(ok: np.ndarray, w: int = W) -> np.ndarray:
    """ok(T×N bool) → 각 t에서 [t-w+1, t] 창 안의 True 개수 (t < w-1은 0)."""
    cs = np.vstack([np.zeros((1, ok.shape[1]), dtype=np.int64), np.cumsum(ok, axis=0)])
    out = np.zeros(ok.shape, dtype=np.int64)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def build_samples(mask: np.ndarray, filled_ok: np.ndarray, times: pd.DatetimeIndex, h: int,
                  bounds: dict[str, tuple[int, int]], w: int = W) -> pd.DataFrame:
    """anchor t별 샘플 표. 분할 소속(경계 걸치면 purged)과 결측 처리 방식별 사용 가능 여부."""
    T, N = mask.shape
    t = np.arange(w - 1, T - h)
    split = np.full(len(t), "purged", dtype=object)
    for name, (s, e) in bounds.items():
        split[(t - w + 1 >= s) & (t + h < e)] = name
    raw_win = window_counts(mask, w)[t]
    fil_win = window_counts(filled_ok, w)[t]
    tgt = mask[t + h]                       # 정답은 항상 원본 관측 여부로 판단(채운 값은 정답으로 쓰지 않음)
    return pd.DataFrame({
        "t_idx": t, "input_start": times[t - w + 1], "input_end": times[t], "target_time": times[t + h],
        "split": split,
        "n_target_observed": tgt.sum(axis=1),
        "strict_raw": (raw_win == w).all(axis=1) & tgt.all(axis=1),
        "strict_locf5": (fil_win == w).all(axis=1) & tgt.all(axis=1),
        "masked_any_target": tgt.any(axis=1),
        "min_input_obs_ratio": (raw_win.min(axis=1) / w).round(4),
    })


def norm_stats(values: np.ndarray, nodes: list[str], train_end: int) -> pd.DataFrame:
    """학습 구간(인덱스 < train_end)의 원본 관측값만으로 노드별 평균·표준편차."""
    tr = values[:train_end]
    return pd.DataFrame({"node_id": nodes, "train_mean": np.nanmean(tr, axis=0), "train_std": np.nanstd(tr, axis=0),
                         "train_n_observed": np.isfinite(tr).sum(axis=0),
                         "train_zero_share": np.nanmean(tr == 0, axis=0).round(4)})


def missing_analysis(values, mask, times, nodes) -> list[str]:
    L = []
    w = L.append
    df = pd.DataFrame(values, index=times, columns=nodes)
    common = ~mask.any(axis=1)
    x = np.diff(np.concatenate([[0], common.astype(np.int8), [0]]))
    w("  [사실] 전 노드 동시 결측 구간: " + ", ".join(
        f"{times[s]:%m-%d %H:%M}~{times[e - 1]:%H:%M} ({e - s}분)"
        for s, e in zip(np.where(x == 1)[0], np.where(x == -1)[0])))
    w("  [추정] 모든 노드가 같은 분에 끊기고 같은 분에 복구 → 개별 센서보다 수집·전송 시스템 측 중단 가능성. 원인 기록 없음.")

    j = nodes.index("23-0009")
    col = mask[:, j]
    x = np.diff(np.concatenate([[1], col.astype(np.int8), [1]]))
    s, e = max(((a, b) for a, b in zip(np.where(x == -1)[0], np.where(x == 1)[0])), key=lambda p: p[1] - p[0])
    before, after = values[s - 1, j], values[e, j] if e < len(values) else np.nan
    others = df.drop(columns="23-0009").iloc[s:e]
    w(f"  [사실] 23-0009 최장 결측: {times[s]} ~ {times[e - 1]} ({e - s:,}분 = {(e - s) / 60:.1f}시간)")
    w(f"         직전 값 {before:.2f} m, 복구 직후 값 {after:.2f} m; 같은 기간 다른 17개 노드는 관측됨"
      f"(결측률 {1 - mask[s:e][:, [k for k in range(len(nodes)) if k != j]].mean():.2%})")
    w(f"         같은 기간 다른 노드 중앙값 수위 최대 {others.median(axis=1).max():.3f} m "
      f"(전체 기간 중앙값 {df.drop(columns='23-0009').median(axis=1).median():.3f} m)")
    w(f"         결측 시작 직전 2시간 23-0009 최대 {np.nanmax(values[max(s - 120, 0):s, j]):.2f} m")
    med = df.drop(columns="23-0009").median(axis=1)
    day = med[times[s].strftime("%Y-%m-%d")]
    w(f"         결측 시작일 다른 노드 중앙값 최고 {day.max():.3f} m at {day.idxmax():%m-%d %H:%M} "
      f"(결측 시작 {times[s]:%H:%M}); 7일 전~시작까지 최고 {med[times[s] - pd.Timedelta('7D'):times[s]].max():.3f} m "
      f"at {med[times[s] - pd.Timedelta('7D'):times[s]].idxmax():%m-%d %H:%M}")
    w("  [추정] 단일 노드만 장기간 끊김 → 해당 센서/통신 장애 가능성(원인 기록 없음). 고수위 기간 전후에 발생했으므로")
    w("         결측이 수위와 무관(MCAR)하다고 가정하지 않는다.")

    j = nodes.index("23-0014")
    v = df["23-0014"]
    w(f"  [사실] 23-0014 0 m 비율 {(v == 0).mean():.1%} (관측값 기준 {(v[mask[:, j]] == 0).mean():.1%}), "
      f"고유값 {v.nunique()}개, 0 초과 값 중앙값 {v[v > 0].median():.3f} m")
    z = (v == 0).astype(int)
    zr = gap_runs(1 - z.to_numpy())  # 0값 연속 구간
    w(f"         0 m 연속 최장 {zr.max() if len(zr) else 0:,}분, 월별 0 비율 "
      + ", ".join(f"{p}: {r:.1%}" for p, r in (v == 0).groupby(v.index.to_period('M')).mean().items()))
    top = df.median(axis=1).resample("D").max().nlargest(3).index
    w("         고수위 상위 3일의 23-0014 0 m 비율: " + ", ".join(
        f"{d:%m-%d} {(v[d.strftime('%Y-%m-%d')] == 0).mean():.1%}" for d in top))
    w("  [추정] 평상시 0 m, 다른 노드가 높을 때만 양수 → 건천 관로/측정 하한 이하 가능성. 시설 위치 대용 좌표이기도 해서")
    w("         센서 설치 위치·0점 기준 확인 필요. 값은 수정하지 않음.")

    w("  [사실] 노드별 결측률 / 결측 구간 수 / 최장(분):")
    for k, n in enumerate(nodes):
        r = gap_runs(mask[:, k])
        w(f"    {n}: {1 - mask[:, k].mean():.4%} / {len(r)} / {int(r.max()) if len(r) else 0:,}")
    return L


def build_report(checks, analysis, code, bounds, times, samples, norm, gap_rule_n) -> str:
    L = []
    w = L.append
    w("[강남] 수위 예측 데이터셋 설계 및 시간순 분할 리포트 (학습 없음)")
    w("=" * 66)
    w("※ 그래프는 도로망 근접 기반 근사이며 실제 하수관로 연결이 아니다.")

    w("\n1. 입력 파일 재검증")
    for k, v in checks.items():
        w(f"  {k}: {v}")

    w("\n2. 결측 분석 ([사실]=데이터로 확인, [추정]=원인 미확인 가설)")
    L.extend(analysis)

    n_obs, n_locf, n_miss = (int((code == c).sum()) for c in (FILL_OBSERVED, FILL_LOCF, FILL_MISSING))
    w("\n3. 결측 처리 비교안 (원본 행렬·마스크는 수정하지 않음, 처리값은 별도 npz)")
    w(f"  A 원본만: 관측 {n_obs:,}칸, 결측 {code.size - n_obs:,}칸 (NaN 유지)")
    w(f"  B 직전값 최대 {LOCF_LIMIT}분 (인과적: 결측 시작 후 {LOCF_LIMIT}분까지만, 미래 정보 없음): "
      f"채움 {n_locf:,}칸, 미처리 결측 {n_miss:,}칸 — 처리코드 0=원본/1=채움/2=미처리")
    w(f"  (참고) '전체 길이 ≤{LOCF_LIMIT}분 구간만 채움' 규칙은 {gap_rule_n:,}칸을 채우지만, 구간이 언제 끝나는지(미래)를")
    w("         알아야 하므로 실시간 예측과 맞지 않아 채택하지 않음. B는 긴 결측의 앞 5분도 채운다.")
    w("  정답(target)에는 채운 값을 절대 사용하지 않는다 — 정답 유효 여부는 항상 원본 mask로 판단.")

    w(f"\n4. 예측 문제: 입력 [t-{W - 1}, t] {W}분 → 정답 t+h (h={HORIZONS}), 18개 노드 동시")
    w("\n5. 시간순 분할 (시간축 인덱스 기준, 무작위 아님)")
    for name, (s, e) in bounds.items():
        w(f"  {name:<5}: idx [{s:,}, {e:,}) = {times[s]} ~ {times[e - 1]} ({e - s:,}분, {(e - s) / len(times):.1%})")
    w(f"  경계 규칙: 샘플 t는 입력 시작 t-{W - 1}과 정답 t+h가 모두 같은 분할 안일 때만 그 분할에 속함. 걸치면 purged(제외).")
    w("\n6. 샘플 수 (결측 처리 방식별)")
    w("  strict_raw   = 18개 노드 입력 60분 전부 원본 관측 + 18개 노드 정답 모두 관측")
    w("  strict_locf5 = 입력 창을 B로 채운 뒤 빈칸 없음 + 정답 모두 원본 관측")
    w("  masked       = 입력 결측 허용(마스크 채널 동반), 정답이 1개 노드 이상 원본 관측인 샘플;"
      " 평가쌍 = 정답이 원본 관측인 (t, 노드)")
    for h, s in samples.items():
        w(f"\n  [h={h}분] 후보 anchor {len(s):,}개")
        g = s.groupby("split")
        tab = pd.DataFrame({
            "샘플": g.size(), "strict_raw": g["strict_raw"].sum(), "strict_locf5": g["strict_locf5"].sum(),
            "masked": g["masked_any_target"].sum(), "평가쌍(t,노드)": g["n_target_observed"].sum(),
        }).reindex(["train", "val", "test", "purged"])
        w(tab.to_string())
        for name in ("train", "val", "test"):
            x = s[s.split == name]
            w(f"    {name}: strict_raw 탈락률 {1 - x.strict_raw.mean():.2%}, "
              f"평가쌍 비율 {x.n_target_observed.sum() / (len(x) * 18):.2%}")

    w("\n7. 기준 모델·GNN 공통 평가 설계")
    w("  - 두 모델 모두 같은 샘플 인덱스 파일(gangnam_forecast_samples_h*.csv)과 같은 평가쌍(정답 원본 관측인 (t, 노드))")
    w("    으로 평가 → 평가 시점·노드 완전 일치")
    w("  - 기준 1 (persistence): ŷ(t+h) = 노드의 마지막 관측값 x(t) (B 처리값). 학습 불필요")
    w("  - 기준 2 (노드별 시계열 모델): 자기 노드 과거 60분만 입력, 이웃 정보 없음 → GNN과 차이는 공간 정보 효과")
    w("  - GNN: 같은 입력 + 그래프(k=3 기본, k=2/k=4/반경 500m 민감도). 그래프만 바꿔도 같은 샘플로 비교 가능")
    w("  - 보조 비교: 간선을 무작위 재배치한 그래프로도 학습해 '공간 구조' 효과와 '파라미터 수' 효과를 구분")

    w("\n8. 정규화 (학습 구간 원본 관측값만, 노드별)")
    w(norm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    w("  검증·테스트 통계는 계산하지 않음. std가 매우 작은 노드(23-0001, 23-0014)는 정규화 후 오차가 확대될 수 있음.")

    w("\n9. 강우 데이터")
    w("  [사실] 저장소·data/raw에 강우/강수량 파일 없음. README 역할표에 '강수량 결합'이 계획 항목으로만 있음.")
    w("  [사실] 수위 기준으로는 노드 다수의 최고수위가 07-17에 집중(이전 리포트). 이것만으로 강우 이벤트 횟수는 단정할 수 없다.")
    w("  → 추가 확보 필요: 기상청 ASOS/AWS 강남·서초 인근 지점 1분/1시간 강수량(2025-06~08).")
    return "\n".join(L) + "\n"


def main() -> None:
    values, mask, times, nodes, checks = load_and_verify()
    filled, code = locf_causal(values)
    np.savez_compressed(OUT_FILLED, values_filled=filled.astype(np.float32), fill_code=code,
                        times=times.to_numpy().astype("datetime64[s]"), node_ids=np.array(nodes, dtype=str))
    bounds = split_bounds(len(times))
    samples = {h: build_samples(mask, np.isfinite(filled), times, h, bounds) for h in HORIZONS}
    for h, s in samples.items():
        s.to_csv(OUT_SAMPLES[h], index=False, encoding="utf-8-sig")
    norm = norm_stats(values, nodes, bounds["train"][1])
    norm.to_csv(OUT_NORM, index=False, encoding="utf-8-sig")
    report = build_report(checks, missing_analysis(values, mask, times, nodes), code, bounds, times, samples,
                          norm, gap_length_rule_fill_count(mask))
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
