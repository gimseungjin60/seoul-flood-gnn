"""강우 이벤트 분포 그림: 강남구 3개소 평균 10분 강우, train/val/test 구간, 유의 이벤트(>=10 mm) 10개 (서울시 강남구 센서 18개 분석 기간).

실행: python -m src.analysis.rain_events_figure
출력: outputs/figures/fig11_rain_events_by_split.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.prepare_rainfall import OUT_EVENTS, OUT_NPZ as RAIN_NPZ, TRAIN_END, VAL_END  # noqa: E402

START = pd.Timestamp("2025-06-01 00:00")
BIG_MM = 50.0


def split_bounds() -> dict:
    return {"train": (START, START + pd.Timedelta(minutes=TRAIN_END)),
            "val": (START + pd.Timedelta(minutes=TRAIN_END), START + pd.Timedelta(minutes=VAL_END)),
            "test": (START + pd.Timedelta(minutes=VAL_END), pd.Timestamp("2025-09-01 00:00"))}


def main() -> None:
    from src.visualization.make_figures import C, _note, _save
    z = np.load(RAIN_NPZ, allow_pickle=True)
    t = pd.DatetimeIndex(z["times"])
    gn = [i for i, u in enumerate(z["gauge_gu"]) if u == "강남구"]
    with np.errstate(all="ignore"):
        rain10 = pd.Series(np.nanmean(z["values"][:, gn], axis=1), index=t)
    rain10 = rain10[(rain10.index >= START) & (rain10.index < pd.Timestamp("2025-09-01"))]
    daily = rain10.resample("D").sum(min_count=1)
    ev = pd.read_csv(OUT_EVENTS, parse_dates=["start", "end"])
    fig, ax = plt.subplots(figsize=(13.3, 4.4))
    tint = {"train": "#E9EFF9", "val": "#FBEFE6", "test": "#E7F2EC"}
    label = {"train": "train 70%", "val": "val 15%", "test": "test 15%"}
    ymax = float(np.nanmax(daily.to_numpy()))
    top = ymax * 1.42
    for name, (a_, b_) in split_bounds().items():
        ax.axvspan(a_, b_, color=tint[name], zorder=0)
        ax.text(a_ + pd.Timedelta(days=1), top * 0.965, label[name], ha="left", va="top", fontsize=13, color=C["ink2"], fontweight="bold")
    ax.bar(daily.index + pd.Timedelta(hours=12), daily.to_numpy(), width=pd.Timedelta(hours=20), color=C["muted"], zorder=3, linewidth=0)
    for r in ev.itertuples():
        seg = daily[(daily.index >= r.start.normalize()) & (daily.index <= r.end.normalize())]
        day = seg.idxmax()
        big = r.total_mm >= BIG_MM
        ax.annotate(f"{int(r.event_id)}\n{r.total_mm:.0f} mm", (day + pd.Timedelta(hours=12), float(seg.max())), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12 if big else 10.5,
                    color=C["s2"] if big else C["ink"], fontweight="bold" if big else "normal", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc=C["surface"], ec=C["s2"] if big else "none", lw=1.3 if big else 0, alpha=0.92))
    ax.set_ylim(0, top)
    ax.set_ylabel("강남구 3개소 평균 일강우 (mm/일)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax.set_xlim(START, pd.Timestamp("2025-09-01"))
    ax.grid(axis="x", visible=False)
    _note(fig, "서울시 강남구 센서 18개 분석 기간(2025-06-01~08-31). 번호=유의 이벤트(총량 10 mm 이상, 6시간 무강우로 분리)의 번호와 총량, 주황=50 mm 이상(2개). test 구간은 진단에 쓰지 않았다.", y=-0.04)
    print(_save(fig, "fig11_rain_events_by_split.png"))


if __name__ == "__main__":
    main()
