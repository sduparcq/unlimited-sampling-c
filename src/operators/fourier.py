import numpy as np

dft_cache = {}


def DFT_matrix(N):
    if N in dft_cache:
        return dft_cache[N]
    else:
        DFT = (1 / np.sqrt(N)) * np.array(
            [[np.exp(-2j * np.pi * i * k / N) for k in range(N)] for i in range(N)]
        )
        dft_cache[N] = DFT
        return DFT
