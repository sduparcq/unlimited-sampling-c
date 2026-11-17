# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# %%
path = "results/dms/param_fitting/L=0.1_zoom_2.csv"


df = pd.read_csv(path)


sigma_values = [i / 600 for i in range(8)]
threshold = 0.5

metrics = [
    ("recovery_rate", "Non-recovery rate"),
    ("uot_q", r"$\overline{UOT}_q$"),
    ("positive_gap", "Mean positive threshold gap"),
    ("negative_gap_rate", "Negative threshold gap rate"),
]


# %%
def compute_metrics(tdf):
    return (
        tdf.groupby(["alpha", "beta"])
        .agg(
            recovery_rate=("is_recovered", "mean"),
            uot_q=("uot_distance_q", "mean"),
            positive_gap=(
                "threshold_gap",
                lambda x: x[x >= 0].mean(),
            ),
            negative_gap_rate=(
                "threshold_gap",
                lambda x: (x < 0).mean(),
            ),
        )
        .reset_index()
    )


# %%
for L, ldf in df.groupby("L", sort=True):
    # Precompute heatmaps for every sigma
    heatmaps = {}

    for sigma in sigma_values:
        sdf = ldf[np.isclose(ldf["sigma"], sigma)]
        tdf = sdf[np.isclose(sdf["threshold"], threshold)]

        if tdf.empty:
            continue

        mdf = compute_metrics(tdf)
        heatmaps[sigma] = {}

        for metric, _ in metrics:
            heatmaps[sigma][metric] = mdf.assign(value=mdf[metric]).pivot(
                index="alpha",
                columns="beta",
                values="value",
            )

    available_sigmas = list(heatmaps.keys())

    if not available_sigmas:
        continue

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(26, 4.5))
    fig.subplots_adjust(
        bottom=0.25,
        top=0.82,
        wspace=0.4,
    )

    images = []
    annotations = []

    first_sigma = available_sigmas[0]

    for j, (metric, title) in enumerate(metrics):
        heatmap = heatmaps[first_sigma][metric]

        im = axes[j].imshow(
            heatmap.values,
            aspect="auto",
            origin="lower",
            vmin=0 if metric in ("recovery_rate", "negative_gap_rate") else None,
            vmax=1 if metric in ("recovery_rate", "negative_gap_rate") else None,
        )

        images.append(im)

        axes[j].set_xticks(range(len(heatmap.columns)))
        axes[j].set_xticklabels([f"{x:.3g}" for x in heatmap.columns])

        axes[j].set_yticks(range(len(heatmap.index)))
        axes[j].set_yticklabels([f"{x:.3g}" for x in heatmap.index])

        axes[j].set_xlabel(r"$\beta$")
        axes[j].set_ylabel(r"$\alpha$")
        axes[j].set_title(title)

        fig.colorbar(
            im,
            ax=axes[j],
            fraction=0.046,
            pad=0.04,
        )

        texts = []
        for y in range(heatmap.shape[0]):
            row = []
            for x in range(heatmap.shape[1]):
                txt = axes[j].text(
                    x,
                    y,
                    "",
                    ha="center",
                    va="center",
                )
                row.append(txt)
            texts.append(row)

        annotations.append(texts)

    axes[0].set_ylabel(rf"$\alpha$" + f"\nthreshold={threshold:.1f}")

    # Slider
    slider_ax = fig.add_axes([0.25, 0.08, 0.5, 0.035])

    slider = Slider(
        ax=slider_ax,
        label=r"$\sigma$",
        valmin=0,
        valmax=len(available_sigmas) - 1,
        valinit=0,
        valstep=1,
        valfmt="%d",
    )

    def update(idx):
        sigma = available_sigmas[int(idx)]

        for j, (metric, _) in enumerate(metrics):
            heatmap = heatmaps[sigma][metric]

            images[j].set_data(heatmap.values)

            if metric not in ("recovery_rate", "negative_gap_rate"):
                images[j].autoscale()

            for y in range(heatmap.shape[0]):
                for x in range(heatmap.shape[1]):
                    value = heatmap.iloc[y, x]

                    annotations[j][y][x].set_text(
                        f"{value:.3f}" if pd.notna(value) else "N/A"
                    )

        fig.suptitle(
            rf"DMS — $L={L}$, $\sigma={sigma:.5f}$",
            fontsize=16,
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    # Keep a reference to the slider
    fig._slider = slider

plt.show()
