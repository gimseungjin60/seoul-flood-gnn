"""2026-06~08 강우 월별 파일(data/raw/rain2026/, gitignore)에서 강남 강우량계 이벤트를 2025와 같은 규칙으로 센다. 읽기 전용.

실행: python -I src/analysis/rain_events_2026.py   (원본이 신뢰할 수 없는 다운로드라 -I 모드, 프로젝트 루트는 아래에서 추가)
결과(2026-06~08): 전체 33개, 유의(>=10 mm) 14개, 50 mm 이상 5개. 마지막 이벤트(08-31)는 자료 끝(23:59)에서 잘려 있어 이어질 수 있음.
"""
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # -I 모드: 프로젝트 루트를 명시적으로만 추가
import numpy as np
import pandas as pd

from src.data.prepare_rainfall import DRY_GAP_STEPS, SIGNIFICANT_MM, find_events

RAW = r"C:\seoul-flood-gnn\data\raw\rain2026"
parts = []
for m in ("202606", "202607", "202608"):
    d = pd.read_csv(f"{RAW}\\rain_{m}.csv", encoding="cp949", dtype=str)
    names = ["gid", "gname", "gu_code", "gu", "gu_cd2", "r10", "ts"] if d.shape[1] == 7 else ["gid", "gname", "gu_code", "gu", "r10", "ts"]
    d.columns = names
    d["src"] = m
    parts.append(d)
d = pd.concat(parts, ignore_index=True)
d["t_raw"] = pd.to_datetime(d["ts"], format="mixed").dt.floor("min")
d["t"] = d["t_raw"].dt.floor("10min") + pd.Timedelta(minutes=9)
d["r"] = pd.to_numeric(d["r10"], errors="raise")
print("행", len(d), "| 지점", d.gid.nunique(), "| 기간", d.t.min(), "~", d.t.max(), "| (gid,t) 중복", int(d.duplicated(["gid", "t"]).sum()),
      "| 이상 시각 행", int(((d.t_raw.dt.minute % 10) != 9).sum()), "| 음수", int((d.r < 0).sum()), "| 최대 10분", d.r.max())
times = pd.date_range("2026-06-01 00:09", "2026-08-31 23:59", freq="10min")
g = d[d.gu == "강남구"]
print("강남 지점:", sorted(g.groupby(["gid", "gname"]).size().to_dict().items()))
piv = g.pivot_table(index="t", columns="gid", values="r", aggfunc="sum").reindex(times)
print("강남 지점별 결측 칸/전체:", piv.isna().sum().to_dict(), len(times))
mean_gn = piv.mean(axis=1, skipna=True).where(piv.notna().any(axis=1))
print("강남 평균 총량(mm):", round(float(mean_gn.sum()), 1), "| 지점별 총량:", piv.sum().round(1).to_dict(), "| 10분 최대(지점):", piv.max().to_dict())
ev = find_events(mean_gn.to_numpy(), times)
sig = ev[ev.total_mm >= SIGNIFICANT_MM].reset_index(drop=True)
print(f"\n전체 이벤트 {len(ev)}개 | 유의(>= {SIGNIFICANT_MM:.0f} mm) {len(sig)}개 | 50 mm 이상 {int((ev.total_mm >= 50).sum())}개")
print(sig[["start", "end", "total_mm", "max_10min_mm", "duration_h"]].to_string(index=True))
daily = mean_gn.resample("D").sum(min_count=1)
print("\n월별 강남 평균 총량(mm):", daily.groupby(daily.index.month).sum().round(1).to_dict())
print("일 강우 상위 8일(mm):", daily.sort_values(ascending=False).head(8).round(1).to_dict())
# 2025와 같은 규칙인지 확인용: 2025 결과(기존 파일)와 같은 함수/규칙을 썼음을 표시
print("\n규칙: 강남 3개소 평균, 6시간(%d스텝) 무강우 분리, 유의 >= %.0f mm" % (DRY_GAP_STEPS, SIGNIFICANT_MM))
