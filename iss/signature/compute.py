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
        for i in range(len(exps[k])):
            if exps[k, i] != 0:
                tmp = tmp * (Z[:, i] ** exps[k, i])
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
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
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k in range(len(exps)):
        for i in range(len(exps[k])):
            if exps[k, i] != 0:
                tmp = tmp * (Z[:, i] ** exps[k, i])
        tmp = np.cumsum(tmp)
        result[k] = tmp
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    return result


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f4[:], f4[:])",
    fastmath=True,
    cache=True,
)
def _exp_iterated_sums_compiled(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
            tmp = tmp * np.exp(-alpha[k] * time)
    tmp = tmp * np.exp(-alpha[-1] * time)
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f4[:])",
    fastmath=True,
    cache=True,
)
def _partial_exp_iterated_sums_compiled(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        result[k, :] = tmp * np.exp(-alpha[k] * time)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
            tmp = tmp * np.exp(-alpha[k] * time)
    return result
