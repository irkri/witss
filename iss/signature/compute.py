import numba
import numpy as np


# --- REALS ---


@numba.njit(
    "f8[:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _reals(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
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
def _partial_reals(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = np.cumsum(tmp)
        result[k] = tmp
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    return result


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
)
def _exp_reals(
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
        if k > 0:
            tmp = tmp * np.exp(-alpha[k-1] * time)
        if k < len(exps) - 1:
            tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    return tmp


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
)
def _exp_outer_reals(
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
        if k > 0:
            tmp = tmp * np.exp(-alpha[k-1] * time)
        tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    tmp = tmp * np.exp(-alpha[-1] * time)
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
)
def _partial_exp_reals(
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
        if k > 0:
            tmp = tmp * np.exp(-alpha[k-1] * time)
        result[k, :] = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    return result


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
)
def _partial_exp_outer_reals(
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
        if k > 0:
            tmp = tmp * np.exp(-alpha[k-1] * time)
        tmp = tmp * np.exp(alpha[k] * time)
        tmp = np.cumsum(tmp)
        result[k, :] = tmp * np.exp(-alpha[k] * time)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
    return result


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f8[:], i4[:,:], f8[:])",
    fastmath=True,
    cache=True,
)
def _cos_reals(
    X: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    expansion: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[0], ), dtype=np.float64)
    for s in range(expansion.shape[0]):
        tmp = np.ones((X.shape[0], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (X[:, i] ** e)
            if k > 0:
                tmp = tmp * (
                    np.sin(alpha[k-1] * time) ** expansion[s, 4*(k-1)+3]
                )
                tmp = tmp * (
                    np.cos(alpha[k-1] * time) ** expansion[s, 4*(k-1)+4]
                )
            if k < len(exp) - 1:
                tmp = tmp * (np.sin(alpha[k] * time) ** expansion[s, 4*k+1])
                tmp = tmp * (np.cos(alpha[k] * time) ** expansion[s, 4*k+2])
            tmp = np.cumsum(tmp)
            if k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
        result += expansion[s, 0] * tmp
    return result


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f8[:], i4[:,:], f8[:])",
    fastmath=True,
    cache=True,
)
def _cos_outer_reals(
    X: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    expansion: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[0], ), dtype=np.float64)
    for s in range(expansion.shape[0]):
        tmp = np.ones((X.shape[0], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (X[:, i] ** e)
            if k > 0:
                tmp = tmp * (
                    np.sin(alpha[k-1] * time) ** expansion[s, 4*(k-1)+3]
                )
                tmp = tmp * (
                    np.cos(alpha[k-1] * time) ** expansion[s, 4*(k-1)+4]
                )
            tmp = tmp * (np.sin(alpha[k] * time) ** expansion[s, 4*k+1])
            tmp = tmp * (np.cos(alpha[k] * time) ** expansion[s, 4*k+2])
            tmp = np.cumsum(tmp)
            if k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
        tmp = tmp * (np.sin(alpha[-1] * time)**expansion[s, -2])
        tmp = tmp * (np.cos(alpha[-1] * time)**expansion[s, -1])
        result += expansion[s, 0] * tmp
    return result


# --- ARCTIC ---


@numba.njit(
    "f8[:](f8[:])",
    fastmath=True,
    cache=True,
)
def cummax(x):
    rmax = x[0]
    y = np.empty_like(x)
    for i, val in enumerate(x):
        if val > rmax: rmax = val
        y[i] = rmax
    return y


@numba.njit(
    "f8[:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _arctic(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    tmp = np.zeros((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp + (Z[:, i] * e)
        tmp = cummax(tmp)
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _partial_arctic(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.zeros((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp + (Z[:, i] * e)
        tmp = cummax(tmp)
        result[k] = tmp
    return result
