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
    """
    ...


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
