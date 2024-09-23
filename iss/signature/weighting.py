from typing import Sequence

import numpy as np


class Weighting:
    ...


class Exponential(Weighting):

    def __init__(self, alpha: Sequence[float] | np.ndarray) -> None:
        self._alpha = alpha

    @property
    def alpha(self) -> np.ndarray:
        return np.array(self._alpha, dtype=np.float32)

    def time(self, T: int) -> np.ndarray:
        return np.arange(1, T+1, dtype=np.float32) / T


class Cosine(Weighting):

    def __init__(self, alpha: Sequence[float]) -> None:
        self._alpha = alpha

    def numpy(self) -> np.ndarray:
        return np.array(self._alpha)
