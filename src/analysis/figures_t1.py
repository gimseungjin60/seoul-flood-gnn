"""시공간 GNN 단기 예측(T1) 그림 2장. 수치는 판정 보고서·집계 파일의 값만 읽는다(새로 계산하지 않는다).

실행: python -m src.analysis.figures_t1
출력: outputs/figures/slide/{s_t1_diff,s_t1_diff_followup,s_t1_warning}.png
- 선형(M2) 대비 오차 차이와 95% CI: F1(docs/19), A1(docs/29), D1(docs/28), N2(docs/31), O1(docs/33) 판정 줄
- 30분 전 큰 상승(200 mm) 첫 도달 맞힘 건수: docs/25 집계(M2, G-road)와 docs/33 보고서(M2F)
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.slide_figures import FS, _save  # noqa: E402
from src.data.clean_nodes import PROCESSED  # noqa: E402
from src.visualization.make_figures import C  # noqa: E402

NUM = r"([+-]?\d+\.\d+)"
DIFF_ROWS = [  # (보고서, 판정 태그, 라벨)
    ("gangnam2026_t1_st_report.txt", "F1", "30분 뒤 · 시공간 GNN"),
    ("gangnam2026_t1_adapt_report.txt", "A1", "30분 뒤 · 연결을 학습하는 GNN"),
    ("gangnam2026_t1_long_report.txt", "D1", "60분 뒤 · 시공간 GNN"),
    ("gangnam2026_t1_nbr_report.txt", "N2", "60분 뒤 · 이웃 연결을 끊은 GNN"),
    ("gangnam2026_t1_oracle_report.txt", "O1", "30분 뒤 · 선형 + 미래 강우 (상한)"),
]


def verdict_line(fname: str, tag: str) -> tuple[float, float, float]:
    for line in (PROCESSED / fname).read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{tag} .*?: {NUM} mm \[{NUM}, {NUM}\]", line)
        if m:
            return tuple(float(x) for x in m.groups())
    raise ValueError(f"{fname}: {tag} 판정 줄이 없다")


def t1_diff(rows=DIFF_ROWS, name: str = "s_t1_diff"):
    """rows: DIFF_ROWS 의 부분집합. 11주차 일지는 10주차 일지와 겹치는 F1 줄을 뺀 판(s_t1_diff_followup)을 쓴다."""
    vals = [verdict_line(f, t) for f, t, _ in rows]
    fig, ax = plt.subplots(figsize=(10.4, 1.0 * len(rows) + 1.0), dpi=130)
    for i, (v, lo, hi) in enumerate(vals):
        col = "#2A78D6" if lo <= 0 <= hi else "#E8692E"
        ax.plot([lo, hi], [i, i], color=col, lw=5, solid_capstyle="round", zorder=3)
        ax.scatter([v], [i], s=160, color=col, edgecolor="white", linewidth=2, zorder=4)
        ax.text(v, i - 0.22, f"{v:+.2f}", ha="center", va="bottom", fontsize=FS, fontweight="bold", color=C["ink"])
    ax.axvline(0, color=C["ink"], ls="--", lw=1.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([lab for *_, lab in rows], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlim(-6.5, 1.8)
    ax.set_xlabel("강우+자기이력 선형 대비 오차 차이 (mm) · 선은 95% 범위\n0보다 왼쪽이면 오차가 작음 · 주황 = 0을 벗어남")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, name)


def t1_warning():
    w = pd.read_csv(PROCESSED / "gangnam2026_t1_warning_counts.csv")
    w = w[(w.horizon_min == 30) & (w.thresh_mm == 200)].set_index("model")
    txt = (PROCESSED / "gangnam2026_t1_oracle_report.txt").read_text(encoding="utf-8")
    sec = txt.split("[판정: 30분 뒤, 200 mm 첫 도달 감지]")[1]
    m = re.search(r"M2F \+미래 강우\s+CSI \S+\s+맞힘 (\d+)/(\d+)", sec)
    rows = [("직전 값 그대로", w.loc["persist", "hits"], w.loc["persist", "reach"]),
            ("시공간 GNN", w.loc["gnn_road", "hits"], w.loc["gnn_road", "reach"]),
            ("강우+자기이력 선형", w.loc["ar_rain", "hits"], w.loc["ar_rain", "reach"]),
            ("선형 + 미래 강우 (상한)", int(m.group(1)), int(m.group(2)))]
    fig, ax = plt.subplots(figsize=(10.4, 4.2), dpi=130)
    for i, (lab, h, n) in enumerate(rows):
        col = "#E8692E" if i == len(rows) - 1 else "#9AA5B5"
        ax.barh(i, h, height=0.55, color=col, zorder=3)
        ax.text(h + 0.6, i, f"{int(h)} / {int(n)}건", va="center", fontsize=FS + 1, fontweight="bold", color=C["ink"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)
    ax.set_xlabel("30분 전에 맞힌 큰 상승(200 mm 이상) 첫 도달 건수")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "s_t1_warning")


if __name__ == "__main__":
    print(t1_diff())
    print(t1_diff([r for r in DIFF_ROWS if r[1] != "F1"], "s_t1_diff_followup"))
    print(t1_warning())
