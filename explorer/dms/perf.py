# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# %%
path = "results/dms/perf/L=0.1_perf_noise.csv"

path = "results/dms/perf/L=0.1_comparison-w-og.csv"

df = pd.read_csv(path)

threshold_values = np.sort(df["threshold"].unique())


def fu(serie):
    return (serie > 0).mean()


def fu2(serie):
    return serie[(serie > 0)].mean()


def compute_mean_df(threshold):
    sdf = df[np.isclose(df["threshold"], threshold)]

    return (
        sdf.groupby("sigma")
        .agg(
            recovery_rate=("is_recovered", "mean"),
            uot_q=("uot_distance_q", "mean"),
            th=("threshold_gap", fu2),
            np=("threshold_gap", fu),
        )
        .reset_index()
        .sort_values("sigma")
    )


# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(bottom=0.18, hspace=0.35, wspace=0.3)

axes = axes.ravel()

metrics = [
    ("recovery_rate", "Recovery rate"),
    ("uot_q", "UOT distance"),
    ("th", "Mean threshold gap"),
    ("np", "Positive threshold gap rate"),
]

lines = []

mean_df = compute_mean_df(threshold_values[0])

for ax, (col, title) in zip(axes, metrics):
    (line,) = ax.plot(
        mean_df["sigma"],
        mean_df[col],
        marker="o",
    )
    ax.set_title(title)
    ax.set_xlabel("Noise σ")
    ax.grid(alpha=0.3)
    if col == "recovery_rate":
        ax.set_ylim(-0.05, 1.05)
    lines.append(line)

# Slider
slider_ax = fig.add_axes((0.2, 0.06, 0.6, 0.03))

slider = Slider(
    ax=slider_ax,
    label="Threshold",
    valmin=0,
    valmax=len(threshold_values) - 1,
    valinit=0,
    valstep=1,
    valfmt="%d",
)

value_text = fig.text(
    0.5,
    0.115,
    f"Threshold = {threshold_values[0]:.4f}",
    ha="center",
)


def update(val):
    idx = int(slider.val)
    threshold = threshold_values[idx]

    mean_df = compute_mean_df(threshold)

    for ax, line, (col, _) in zip(axes, lines, metrics):
        line.set_data(mean_df["sigma"], mean_df[col])
        ax.autoscale_view()
        if col == "recovery_rate":
            ax.set_ylim(-0.05, 1.05)

    value_text.set_text(f"Threshold = {threshold:.4f}")
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()
