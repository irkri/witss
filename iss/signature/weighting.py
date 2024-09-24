from typing import Sequence
from itertools import product

import numpy as np

from ..words.word import Word


class Weighting:
    ...


class Exponential(Weighting):

    def __init__(self, alpha: Sequence[float] | np.ndarray) -> None:
        self._alpha = alpha

    @property
    def alpha(self) -> np.ndarray:
        return np.array(self._alpha, dtype=np.float64)

    def time(self, T: int) -> np.ndarray:
        return np.arange(1, T+1, dtype=np.float64) / T


class Cosine(Weighting):

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