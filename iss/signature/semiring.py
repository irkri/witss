from enum import Enum


class Semiring:
    """A semiring is the space over which operations of an iterated sum
    are defined. Changing the semiring for the ISS changes the behavior
    of the construct.
    """
    ...


class Reals(Semiring):
    """The semiring of real numbers with default addition and
    multiplication is the standard space of the ISS.

    Args:
        normalize (bool, optional): If set to true, normalizes the
            iterated sums by normalizing each cumulative sum. This
            normalization leads to computing averages instead of sums.
            Normalization often prevents overflow. Defaults to False.
    """

    def __init__(self, normalize: bool = False) -> None:
        self._normalize = normalize

    @property
    def normalized(self) -> bool:
        return self._normalize


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
