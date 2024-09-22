import numba
import numpy as np


@numba.njit(
    "f8[:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _iterated_sums_compiled(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k in range(len(exps)):
        C = np.ones((Z.shape[0], ), dtype=np.float64)
        for i in range(len(exps[k])):
            C = C * (Z[:, i] ** exps[k, i])
            # if exps[k][i] > 0:
            #     for _ in range(exps[k][i]):
            #         C = C * Z[:, i]
            # elif exps[k][i] < 0:
            #     for _ in range(-exps[k][i]):
            #         C = C / Z[:, i]
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp[k:] = tmp[k:] * C[k:]
        tmp[k:] = np.cumsum(tmp[k:])
        # if k > 0:
        #     tmp = tmp * np.exp(- weights * scalar[k-1])
        # if k < len(exps) - 1:
            # tmp = tmp * np.exp(weights * scalar[k])
            # tmp[k:] = np.cumsum(tmp[k:])
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _partial_iterated_sums_compiled(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[1], ), dtype=np.float64)
    tmp = np.ones((Z.shape[1], ), dtype=np.float64)
    for k in range(len(exps)):
        C = np.ones((Z.shape[1], ), dtype=np.float64)
        for i in range(len(exps[k])):
            C = C * (Z[i, :] ** exps[k][i])
            # if exps[k][i] > 0:
            #     for _ in range(occurence):
            #         C = C * Z[letter, :]
            # elif exps[k][i] < 0:
            #     for _ in range(-occurence):
            #         C = C / Z[letter, :]
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp[k:] = tmp[k:] * C[k:]
        # if k > 0:
        #     tmp = tmp * np.exp(- weights * scalar[k-1])
        result[k, k:] = np.cumsum(tmp[k:])
        # if k < len(exps) - 1:
            # tmp = tmp * np.exp(weights * scalar[k])
            # tmp[k:] = np.cumsum(tmp[k:])
    return result
