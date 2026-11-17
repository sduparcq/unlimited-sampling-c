import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "results/fista/perf/test1.csv"

path = "results/fista/perf/test1_npt.csv"

path = "results/fista/perf/test_pt_high_sigma.csv"

df = pd.read_csv(path)

mean_df = (
    df.groupby("sigma")
    .agg(
        recovery_rate=("is_recovered", "mean"),
        uot=("uot_distance", "mean"),
        uot_q=("uot_distance_q", "mean"),
    )
    .reset_index()
    .sort_values("sigma")
)

plt.subplot(3, 1, 1)
plt.plot(mean_df["sigma"], mean_df["recovery_rate"])
plt.subplot(3, 1, 2)
plt.plot(mean_df["sigma"], mean_df["uot_q"])
plt.subplot(3, 1, 3)
plt.plot(mean_df["sigma"], mean_df["uot"])

plt.show()
