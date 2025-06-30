from typing import Literal

import numpy as np


class Semiring:
    """The abstract semiring class. Should not be used as an actual
    semiring.

    A semiring is the space over which operations of an iterated sum are
    defined. Changing the semiring for the ISS changes the behavior of
    the construct.
    """
    ...


class Reals(Semiring):
    """The semiring of real numbers with default addition and
    multiplication is the standard space of the ISS.

    Args:
        normalize (str, optional): Normalizes the iterated sums by
            dividing each cumulative sum by some time series. This time
            series can either be ``np.arange(1, T+1)`` for
            ``normalize="linear"`` or ``np.sqrt(np.arange(1, T+1))`` for
            ``normalize="sqrt"``. Defaults to no normalization.
    """

    def __init__(
        self,
        normalize: Literal["sqrt", "linear"] | None = None,
    ) -> None:
        self._normalize = normalize

    def normalization(self, T: int) -> np.ndarray:
        if self._normalize is None:
            return np.ones(T, dtype=np.float64)
        if self._normalize == "linear":
            return np.arange(1, T+1, dtype=np.float64)
        if self._normalize == "sqrt":
            return np.sqrt(np.arange(1, T+1, dtype=np.float64))
        else:
            raise ValueError(f"Normalization {self._normalize} does not exist")


class Arctic(Semiring):
    """The arctic semiring has ``max`` as additive operation and
    standard addition as multiplicative operation.
    """
    ...


class Bayesian(Semiring):
    """The Bayesian semiring has ``max`` as additive operation and
    standard multiplication as multiplicative operation. Only positive
    real numbers are allowed.
    """
    ...
