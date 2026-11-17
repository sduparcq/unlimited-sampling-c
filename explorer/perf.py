"""Compare OG and DMS performance; run with `python explorer/perf.py`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider


ROOT = Path(__file__).resolve().parents[1]
DMS_PATH = ROOT / "results/dms/perf/L=0.1_comparison-w-og.csv"
OG_PATH = ROOT / "results/og/perf/L=0.1-for-comparison-w-dms.csv"


def compute_dms_means(df, threshold):
    selected = df[df["threshold"] == threshold]
    return selected.groupby("sigma").agg(
        recovery_rate=("is_recovered", "mean"),
        uot_q=("uot_distance_q", "mean"),
        threshold_gap=("threshold_gap", lambda values: values[values > 0].mean()),
        positive_gap_rate=("threshold_gap", lambda values: (values > 0).mean()),
    ).sort_index()


def create_figure(dms_df, og_df):
    """Return the figure and slider (keep the slider alive for interaction)."""
    thresholds = np.sort(dms_df["threshold"].dropna().unique())
    if not len(thresholds):
        raise ValueError("DMS results contain no threshold values.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(bottom=0.19, top=0.88, hspace=0.4, wspace=0.3)
    fig.suptitle("OG and DMS performance across noise levels", fontsize=16)
    metrics = [
        ("recovery_rate", "Recovery rate: OG vs DMS"),
        ("uot_q", "Mean UOT distance: OG vs DMS"),
        ("threshold_gap", "DMS: mean positive threshold gap"),
        ("positive_gap_rate", "DMS: positive threshold gap rate"),
    ]

    # OG curves remain fixed as the DMS threshold changes.
    for index, (np_value, group) in enumerate(og_df.groupby("Np", sort=True)):
        means = group.groupby("sigma").agg(
            recovery_rate=("is_recovered", "mean"),
            uot_q=("uot_distance_q", "mean"),
        ).sort_index()
        for ax, (metric, _) in zip(axes[0], metrics[:2]):
            ax.plot(
                means.index, means[metric], linestyle="--", marker=".",
                color=f"C{index % 10}", label=f"OG (Np = {np_value})", alpha=0.8,
            )

    means = compute_dms_means(dms_df, thresholds[0])
    dms_lines = []
    for ax, (metric, title) in zip(axes.flat, metrics):
        (line,) = ax.plot(
            means.index, means[metric], color="black", marker="o",
            linewidth=2, label="DMS (selected threshold)",
        )
        dms_lines.append(line)
        ax.set_title(title)
        ax.set_xlabel("Noise σ")
        ax.grid(alpha=0.3)
        if metric in ("recovery_rate", "positive_gap_rate"):
            ax.set_ylim(-0.05, 1.05)
    for ax in axes[0]:
        ax.legend(fontsize="small")

    slider_ax = fig.add_axes((0.2, 0.06, 0.6, 0.03))
    slider = Slider(
        ax=slider_ax, label="DMS threshold", valmin=0,
        valmax=max(1, len(thresholds) - 1), valinit=0,
        valstep=np.arange(len(thresholds)),
    )
    slider.valtext.set_text(f"{thresholds[0]:.6g}")
    if len(thresholds) == 1:
        slider.set_active(False)
    value_text = fig.text(
        0.5, 0.12, f"DMS threshold = {thresholds[0]:.6g}", ha="center",
    )

    def update(value):
        threshold = thresholds[int(value)]
        means = compute_dms_means(dms_df, threshold)
        for ax, line, (metric, _) in zip(axes.flat, dms_lines, metrics):
            line.set_data(means.index, means[metric])
            ax.relim()
            ax.autoscale_view()
        slider.valtext.set_text(f"{threshold:.6g}")
        value_text.set_text(f"DMS threshold = {threshold:.6g}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    return fig, slider


if __name__ == "__main__":
    fig, slider = create_figure(pd.read_csv(DMS_PATH), pd.read_csv(OG_PATH))
    plt.show()
