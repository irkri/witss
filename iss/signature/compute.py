import numpy as np
import numba


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f8[:], i8)",
    fastmath=True,
    cache=True,
)
def _iterated_sums_compiled(
    Z: np.ndarray,
    dims: np.ndarray,
    exps: np.ndarray,
    scalar: np.ndarray,
    weights: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.zeros((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones((Z.shape[1], ), dtype=np.float64)
    for k in range(len(dims)):
        C = np.ones((Z.shape[1], ), dtype=np.float64)
        for i in range(len(dims[k])):
            if occurence > 0:
                for _ in range(occurence):
                    C = C * Z[letter, :]
            elif occurence < 0:
                for _ in range(-occurence):
                    C = C / Z[letter, :]
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp[k:] = tmp[k:] * C[k:]
        if k > 0:
            tmp = tmp * np.exp(- weights * scalar[k-1])
        if len(word) - k <= extended:
            result[extended-(len(word)-k), k:] = np.cumsum(tmp[k:])
        if k < len(word) - 1:
            tmp = tmp * np.exp(weights * scalar[k])
            tmp[k:] = np.cumsum(tmp[k:])
    return result
