# %%
from src.signals.factory import SignalFactory
from src.signals.signal import SignalParameters
import matplotlib.pyplot as plt

# %%

sp = SignalParameters(
    N=100,
    L=1 / 150,
    Omega=10.0,
    bounds=(0, 1),
    M=0.05,
    sigma=0.03,
    pt=False,
    s_seed=1,
    n_seed=1,
)

sf = SignalFactory()

# %%
sps = [
    SignalParameters(
        N=500,
        L=0.1,
        Omega=i * 10.0,
        bounds=(0, 1),
        M=0.05,
        sigma=1 / 150,
        pt=False,
        s_seed=i,
        n_seed=1,
    )
    for i in range(1, 2)
]
# %%
for sp in sps:
    s = sf.generate_signal(sp)

    plt.plot(s.support, s.signal)
    plt.plot(s.support, s.mod_samples, ls="--")

    plt.plot(s.support, s.nms, ls="--")
    plt.show()
