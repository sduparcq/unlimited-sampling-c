import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "results/dms/param_fitting/new.csv"

path = "results/dms/param_fitting/L=0.1_zoom_2.csv"

df = pd.read_csv(path)

sigma_values = [0, 0.03, 0.05]
threshold_values = [0.0, 0.5]

df = df[df["threshold"].isin(threshold_values)]
df = df[df["sigma"].isin(sigma_values)]


metrics = [
    ("recovery_rate", "Non-recovery rate"),
    ("uot_q", r"$\overline{UOT}_q$"),
    ("positive_gap", "Mean positive threshold gap"),
    ("negative_gap_rate", "Negative threshold gap rate"),
]

for sigma in sigma_values:
    sdf = df[np.isclose(df["sigma"], sigma)]

    fig, axes = plt.subplots(
        nrows=len(threshold_values),
        ncols=4,
        figsize=(26, 2.5 * len(threshold_values)),
    )

    for i, threshold in enumerate(threshold_values):
        tdf = sdf[np.isclose(sdf["threshold"], threshold)]

        mdf = (
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

        for j, (metric, title) in enumerate(metrics):
            if metric == "recovery_rate":
                values = 1 - mdf[metric]
            else:
                values = mdf[metric]

            heatmap = mdf.assign(value=values).pivot(
                index="alpha",
                columns="beta",
                values="value",
            )

            im = axes[i, j].imshow(
                heatmap.values,
                aspect="auto",
                origin="lower",
                vmin=0 if metric in ("recovery_rate", "negative_gap_rate") else None,
                vmax=1 if metric in ("recovery_rate", "negative_gap_rate") else None,
            )

            axes[i, j].set_xticks(range(len(heatmap.columns)))
            axes[i, j].set_xticklabels([f"{x:.3g}" for x in heatmap.columns])

            axes[i, j].set_yticks(range(len(heatmap.index)))
            axes[i, j].set_yticklabels([f"{x:.3g}" for x in heatmap.index])

            axes[i, j].set_xlabel(r"$\beta$")
            axes[i, j].set_ylabel(r"$\alpha$")
            axes[i, j].set_title(title)

            fig.colorbar(
                im,
                ax=axes[i, j],
                fraction=0.046,
                pad=0.04,
            )

            for y in range(heatmap.shape[0]):
                for x in range(heatmap.shape[1]):
                    value = heatmap.iloc[y, x]

                    axes[i, j].text(
                        x,
                        y,
                        f"{value:.3f}" if pd.notna(value) else "N/A",
                        ha="center",
                        va="center",
                    )

        axes[i, 0].set_ylabel(rf"$\alpha$" + f"\nthreshold={threshold:.1f}")

    fig.suptitle(
        rf"DMS — $\sigma={sigma}$",
        fontsize=16,
    )

    fig.tight_layout()
    plt.show()
