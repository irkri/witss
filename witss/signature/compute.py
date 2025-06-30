import numba
import numpy as np


# --- REALS ---


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f8[:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _reals(
    Z: np.ndarray,
    exps: np.ndarray,
    div: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.empty((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            tmp = np.cumsum(tmp)
            if k == 0 or not strict:
                tmp = tmp / div  # type: ignore
            else:
                tmp[k:] = tmp[k:] / div[:-k]  # type: ignore
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
        result[n, :] = tmp
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f8[:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _partial_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    div: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            tmp = np.cumsum(tmp)
            if k == 0 or not strict:
                tmp = tmp / div  # type: ignore
            else:
                tmp[k:] = tmp[k:] / div[:-k]  # type: ignore
            result[k, n] = tmp
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
    return result


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f8[:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _exp_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.empty((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            if k > 0:
                tmp = tmp * np.exp(-alpha[k-1] * time)
            if k < len(exps) - 1:
                tmp = tmp * np.exp(alpha[k] * time)
            tmp = np.cumsum(tmp)
            if k == 0:
                tmp = tmp / div
            else:
                tmp[k:] = tmp[k:] / div[:-k]
            if k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
        result[n] = tmp
    return result


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f8[:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _exp_outer_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.empty((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            if k > 0:
                tmp = tmp * np.exp(-alpha[k-1] * time)
            tmp = tmp * np.exp(alpha[k] * time)
            tmp = np.cumsum(tmp)
            if k == 0:
                tmp = tmp / div  # type: ignore
            else:
                tmp[k:] = tmp[k:] / div[:-k]  # type: ignore
            if k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
        tmp = tmp * np.exp(-alpha[-1] * time)
        result[n] = tmp
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f8[:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _partial_exp_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            if k > 0:
                tmp = tmp * np.exp(-alpha[k-1] * time)
            result[k, n, :] = np.cumsum(tmp)
            if k == 0:
                result[k, n] = result[k, n] / div
            else:
                result[k, n, k:] = result[k, n, k:] / div[:-k]
            if k < len(exps) - 1:
                tmp = tmp * np.exp(alpha[k] * time)
                tmp = np.cumsum(tmp)
                if k == 0:
                    tmp = tmp / div
                else:
                    tmp[k:] = tmp[k:] / div[:-k]
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f8[:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _partial_exp_outer_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            if k > 0:
                tmp = tmp * np.exp(-alpha[k-1] * time)
            tmp = tmp * np.exp(alpha[k] * time)
            tmp = np.cumsum(tmp)
            if k == 0:
                tmp = tmp / div
            else:
                tmp[k:] = tmp[k:] / div[:-k]
            result[k, n, :] = tmp * np.exp(-alpha[k] * time)
            if k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
    return result


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f8[:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _cos_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    expansion: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.zeros((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        for s in range(expansion.shape[0]):
            tmp = np.ones((Z.shape[1], ), dtype=np.float64)
            for k, exp in enumerate(exps):
                for i, e in enumerate(exp):
                    if e != 0:
                        tmp = tmp * (Z[n, :, i] ** e)
                if k > 0:
                    tmp = tmp * (
                        np.sin(alpha[k-1] * time) ** expansion[s, 4*(k-1)+3]
                    )
                    tmp = tmp * (
                        np.cos(alpha[k-1] * time) ** expansion[s, 4*(k-1)+4]
                    )
                if k < len(exp) - 1:
                    tmp = tmp * (np.sin(alpha[k]*time) ** expansion[s, 4*k+1])
                    tmp = tmp * (np.cos(alpha[k]*time) ** expansion[s, 4*k+2])
                tmp = np.cumsum(tmp)
                if k == 0:
                    tmp = tmp / div  # type: ignore
                else:
                    tmp[k:] = tmp[k:] / div[:-k]  # type: ignore
                if k < len(exps) - 1:
                    tmp = np.roll(tmp, 1)
                    tmp[0] = 0
            result[n, :] += expansion[s, 0] * tmp
    return result


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f8[:], i4[:,:], f8[:], f8[:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _cos_outer_reals(
    Z: np.ndarray,
    exps: np.ndarray,
    alpha: np.ndarray,
    expansion: np.ndarray,
    time: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    result = np.zeros((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        for s in range(expansion.shape[0]):
            tmp = np.ones((Z.shape[1], ), dtype=np.float64)
            for k, exp in enumerate(exps):
                for i, e in enumerate(exp):
                    if e != 0:
                        tmp = tmp * (Z[n, :, i] ** e)
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
                if k == 0:
                    tmp = tmp / div  # type: ignore
                else:
                    tmp[k:] = tmp[k:] / div[:-k]  # type: ignore
                if k < len(exps) - 1:
                    tmp = np.roll(tmp, 1)
                    tmp[0] = 0
            tmp = tmp * (np.sin(alpha[-1] * time)**expansion[s, -2])
            tmp = tmp * (np.cos(alpha[-1] * time)**expansion[s, -1])
            result[n, :] += expansion[s, 0] * tmp
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
    "f8[:,:](f8[:,:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _arctic(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.empty((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.zeros((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp + (Z[n, :, i] * e)
            tmp = cummax(tmp)
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = - np.inf
        result[n, :] = tmp
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _partial_arctic(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.zeros((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp + (Z[n, :, i] * e)
            tmp = cummax(tmp)
            result[k, n] = tmp
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = - np.inf
    return result


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
)
def _arctic_argmax(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.zeros((1, Z.shape[0]), dtype=np.float64)
    indices = np.zeros((len(exps), Z.shape[0]), dtype=np.int32)
    for k, exp in enumerate(exps):
        for i, e in enumerate(exp):
            if e != 0:
                result[0, :] = result[0, :] + (Z[:, i] * e)
        for t in range(Z.shape[0]-1):
            if result[0, t+1] < result[0, t]:
                result[0, t+1] = result[0, t]
                indices[k, t+1] = indices[k, t]
            else:
                indices[k, t+1] = t+1
        if strict and k < len(exps) - 1:
            result[0, :] = np.roll(result[0, :], 1)
            result[0, 0] = - np.inf
            indices[k, :] = np.roll(indices[k, :], 1)
            indices[k, 0] = 0
    for s in range(len(exps)-1, 0, -1):
        indices[s-1, :] = indices[s-1, indices[s]]
    return np.concatenate((result, indices))


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
)
def _partial_arctic_argmax(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0]), dtype=np.float64)
    indices = np.zeros(
        (int((len(exps) * (len(exps)+1)/2)), Z.shape[0]),
        dtype=np.int32,
    )
    for k, exp in enumerate(exps):
        if k > 0:
            if strict:
                result[k] = np.roll(result[k-1], 1)
                result[k, 0] = - np.inf
                for l in range(k, len(exps)):
                    c = int(l*(l+1)/2)+k
                    indices[c, :] = np.roll(indices[c, :], 1)
                    indices[c, 0] = 0
            else:
                result[k] = result[k-1]
        for i, e in enumerate(exp):
            if e != 0:
                result[k] = result[k] + (Z[:, i] * e)
        for t in range(Z.shape[0]-1):
            if result[k, t+1] < result[k, t]:
                result[k, t+1] = result[k, t]
                for l in range(k, len(exps)):
                    c = int(l*(l+1)/2)+k
                    indices[c, t+1] = indices[c, t]
            else:
                for l in range(len(exps)-1, k-1, -1):
                    c = int(l*(l+1)/2)+k
                    indices[c, t+1] = t+1
    for k in range(len(exps)-1, -1, -1):
        index = int(k * (k+1) / 2)
        for s in range(k, 0, -1):
            indices[index+s-1, :] = indices[index+s-1, indices[index+s]]
    return np.concatenate((result, indices))


# --- BAYESIAN ---


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _bayesian(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.empty((Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            tmp = cummax(tmp)
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = - np.inf
        result[n, :] = tmp
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], boolean)",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _partial_bayesian(
    Z: np.ndarray,
    exps: np.ndarray,
    strict: bool,
) -> np.ndarray:
    result = np.zeros((len(exps), Z.shape[0], Z.shape[1]), dtype=np.float64)
    for n in numba.prange(Z.shape[0]):
        tmp = np.ones((Z.shape[1], ), dtype=np.float64)
        for k, exp in enumerate(exps):
            for i, e in enumerate(exp):
                if e != 0:
                    tmp = tmp * (Z[n, :, i] ** e)
            tmp = cummax(tmp)
            result[k, n] = tmp
            if strict and k < len(exps) - 1:
                tmp = np.roll(tmp, 1)
                tmp[0] = - np.inf
    return result
