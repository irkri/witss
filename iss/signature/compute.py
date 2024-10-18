import numba
import numpy as np


# --- REALS ---


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
)
def _reals(
    Z: np.ndarray,
    exps: np.ndarray,
    norm: bool,
) -> np.ndarray:
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    if norm:
        div = np.arange(1, Z.shape[0]+1)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = np.cumsum(tmp)
        if k < len(exps) - 1:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
            if norm:
                tmp[k+1:] = tmp[k+1:] / div[:-(k+1)] # type: ignore
    if norm:
        if len(exps) > 1:
            tmp[len(exps)-1:] = (
                tmp[len(exps)-1:] / div[:-len(exps)+1] # type: ignore
            )
        else:
            tmp = tmp / div # type: ignore
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
)
def _partial_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    norm: bool,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    if norm:
        div = np.arange(1, Z.shape[0]+1)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = np.cumsum(tmp)
        if norm:
            tmp = tmp / div
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
def _partial_arctic_argmax(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    result = np.zeros((2, len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.zeros((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp + (Z[:, i] * e)
        result[0, k, 0] = tmp[0]
        for t in range(1, Z.shape[0]):
            if result[0, k, t-1] >= tmp[t]:
                result[0, k, t] = result[0, k, t-1]
                result[1, k, t] = result[1, k, t-1]
            else:
                result[0, k, t] = tmp[t]
                result[1, k, t] = t
        if k < len(exps) - 1:
            tmp = cummax(tmp)
    # translate indices back to their actual position
    n = int(len(exps) + (len(exps) * (len(exps)+1) / 2))
    translated_results = np.zeros((n, Z.shape[0]), dtype=np.float64)
    for k in range(len(exps)-1, -1, -1):
        index = int(k + (k * (k+1) / 2))
        translated_results[index] = result[0, k]
        translated_results[index+k+1] = result[1, k]
        for s in range(k, 0, -1):
            c = int(translated_results[index+s+1, -1])+1
            translated_results[index+s, :c] = result[1, (s-1), :c]
            translated_results[index+s, c:] = result[1, (s-1), c-1]
    return translated_results


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


# --- BAYESIAN ---


@numba.njit(
    "f8[:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _bayesian(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = cummax(tmp)
    return tmp


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _partial_bayesian(
    Z: np.ndarray,
    exps: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    tmp = np.ones((Z.shape[0], ), dtype=np.float64)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                tmp = tmp * (Z[:, i] ** e)
        tmp = cummax(tmp)
        result[k] = tmp
    return result
