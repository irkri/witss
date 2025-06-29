from typing import Sequence
from itertools import product

import numpy as np

from ..words.word import Word


class Weighting:
    """Abstract class to be used as a weighting. May not be used as an
    actual weighting itself.
    """
    ...


class Exponential(Weighting):
    """An exponential weighting for the iterated sum.
    The factors inside an iterated sum with two running indices
    ``t_1<t_2<=t`` for a time series of length ``T`` are:
    ```
        exp(-alpha[1]*(t-t_2)/T) * exp(-alpha[0]*(t_2-t_1)/T)
    ```
    The alpha factors are ordered from inner to outer weighting.

    Args:
        alpha (Sequence[float]): The alpha values for the weighting.
        outer (bool, optional): If set to False, excludes the most outer
            weighting ``exp(-alpha[-1]*(t-t_p)/T)``. One can achieve the
            same by setting the last alpha value to zero, except in
            cases where one wants to also compute partial iterated sums.
    """

    def __init__(
        self,
        alpha: Sequence[float] | np.ndarray,
        outer: bool = True,
    ) -> None:
        self._alpha = alpha
        self._outer = outer

    @property
    def alpha(self) -> np.ndarray:
        return np.array(self._alpha, dtype=np.float64)

    @property
    def outer(self) -> bool:
        return self._outer

    def time(self, T: int) -> np.ndarray:
        return np.arange(1, T+1, dtype=np.float64) / T


class Cosine(Weighting):
    """A cosine weighting for the iterated sum.
    The factors inside an iterated sum with two running indices
    ``t_1<t_2<=t`` are:
    ```
        cos(-alpha[1]*(t-t_2)/T) * cos(-alpha[0]*(t_2-t_1)/T)
    ```
    The alpha factors are ordered from inner to outer weighting.

    Args:
        alpha (Sequence[float]): The alpha values for the weighting.
        outer (bool, optional): If set to False, excludes the most outer
            weighting ``cos(-alpha[-1]*(t-t_p)/T)``. One can achieve the
            same by setting the last alpha value to zero, except in
            cases where one wants to also compute partial iterated sums.
    """

    def __init__(
        self,
        alpha: Sequence[float] | np.ndarray,
        exponent: int = 1,
        outer: bool = True,
    ) -> None:
        self._alpha = alpha
        self._exponent = exponent
        self._outer = outer

    @property
    def alpha(self) -> np.ndarray:
        return np.array(self._alpha, dtype=np.float64)

    @property
    def outer(self) -> bool:
        return self._outer

    def time(self, T: int) -> np.ndarray:
        return np.arange(1, T+1, dtype=np.float64) / T

    def expansion(self, word: Word) -> np.ndarray:
        p = len(word) + 1 if self._outer else len(word)
        trig_id = []
        trig_exp = [self._exponent, 0]
        trig_coeff = 1
        for k in range(self._exponent+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (self._exponent - k) // (k + 1)
        weightings = np.zeros(
            ((self._exponent+1)**(p-1), 4*p-3),
            dtype=np.int32,
        )
        weightings[:, 0] = 1
        for c, comb in enumerate(product(trig_id, repeat=p-1)):
            for i in range(p-1):
                weightings[c, 0] *= int(comb[i][0])
                weightings[c, 4*i+1] += int(comb[i][1])
                weightings[c, 4*i+3] += int(comb[i][1])
                weightings[c, 4*i+2] += int(comb[i][2])
                weightings[c, 4*i+4] += int(comb[i][2])
        return weightings
