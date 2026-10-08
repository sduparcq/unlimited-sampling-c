import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "results/fista/benchmark.csv"

df = pd.read_csv(path)


threshold_values = [0.1, 0.5, 0.9]
sigma_values = [0, 0.03, 0.07]

df = df[df["threshold"].isin(threshold_values)]
df = df[df["sigma"].isin(sigma_values)]

for sigma in sigma_values:
    sdf = df[np.isclose(df["sigma"], sigma)]

    fig, axes = plt.subplots(
        nrows=len(threshold_values),
        ncols=3,
        figsize=(15, 3 * len(threshold_values)),
        sharex=True,
    )

    for i, threshold in enumerate(threshold_values):
        tdf = sdf[np.isclose(sdf["threshold"], threshold)]

        mdf = (
            tdf.groupby("tau")
            .agg(
                recovery_rate=("is_recovered", "mean"),
                uot=("uot_distance", "mean"),
                uot_q=("uot_distance_q", "mean"),
            )
            .reset_index()
            .sort_values("tau")
        )

        axes[i, 0].plot(
            mdf["tau"],
            1 - mdf["recovery_rate"],
            marker="o",
        )
        axes[i, 0].set_ylabel(f"threshold={threshold:.1f}\n% non-recovery")

        axes[i, 1].plot(
            mdf["tau"],
            mdf["uot"],
            marker="o",
        )
        axes[i, 1].set_ylabel(r"$\overline{UOT}$")

        axes[i, 2].plot(
            mdf["tau"],
            mdf["uot_q"],
            marker="o",
        )
        axes[i, 2].set_ylabel(r"$\overline{UOT}_q$")

        for j in range(3):
            axes[i, j].grid(alpha=0.3)

    axes[0, 0].set_title("Non-recovery rate")
    axes[0, 1].set_title(r"$\overline{UOT}$")
    axes[0, 2].set_title(r"$\overline{UOT}_q$")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\tau$")

    fig.suptitle(
        rf"Recovery and UOT metrics — $\sigma={sigma}$",
        fontsize=16,
    )

    fig.tight_layout()

    plt.show()
