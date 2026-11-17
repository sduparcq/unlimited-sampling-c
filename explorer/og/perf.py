# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
path = "results/og/perf/test1.csv"

path = "results/og/perf/L=0.1-for-comparison-w-dms.csv"

df = pd.read_csv(path)
# %%
print(df.columns)
# %%


Ns = [1, 2, 3, 4, 5]
dfs = [df[df["Np"] == n] for n in Ns]

for n, df in zip(Ns, dfs):
    mean_loc_df = (
        df.groupby("sigma")
        .agg(
            recovery_rate=("is_recovered", "mean"),
            uot_q=("uot_distance_q", "mean"),
        )
        .reset_index()
        .sort_values("sigma")
    )
    plt.plot(
        mean_loc_df["sigma"], mean_loc_df["recovery_rate"], label=f"N = {n}", ls="--"
    )
plt.legend()
plt.show()
