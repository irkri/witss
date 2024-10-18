import warnings
from typing import Optional

import numpy as np

from ..words.word import Word
from .compute import (_arctic, _partial_arctic_argmax, _bayesian, _cos_outer_reals,
                      _cos_reals, _exp_outer_reals, _exp_reals,
                      _partial_arctic, _partial_bayesian,
                      _partial_exp_outer_reals, _partial_exp_reals,
                      _partial_reals, _reals)
from .semiring import Arctic, Bayesian, Reals, Semiring
from .weighting import Cosine, Exponential, Weighting


def iss(
    x: np.ndarray,
    word: Word | str,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    semiring: Semiring = Reals(),
    normalize: bool = False,
) -> np.ndarray:
    """Calculate the iterated sums signature of the given time series
    evaluated at the given word.

    Args:
        x (np.ndarray): Input array of 2 dimensions ``(T, d)`` where
            ``T`` is the sequence length and ``d`` the dimension of each
            time step.
        word (Word | str): The word the signature should be evaluated
            on.
        partial (bool): If True, also evaluates the signature for
            all prefix words of the given word, e.g. if
            ``word=[1][2][3]``, the method also returns the signature
            for ``word=[1]`` and ``word=[1][2]``. Defaults to False.
        weighting (Weighting, optional): Weighting for the iterated sum
            that boosts or penalizes distances between time steps.
            Defaults to None.
        semiring (Semiring, optional): Sets the semiring for the
            iterated sum. This changes the behavior of the ISS. Defaults
            to ``Reals``.
        normalize (bool, optional): If True, normalizes the iterated sum
            for the real semiring by normalizing each cumulative sum.
            This prevents overflow. Defaults to False.
    """
    word = word if isinstance(word, Word) else Word(word)

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    type_ = None
    if x.dtype != np.float64:
        type_ = x.dtype
        x = x.astype(np.float64)

    if word.is_empty():
        return np.ones((x.shape[0], ), dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim == 3:
        y = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            y[i] = iss(x[i], word=word, partial=partial, weighting=weighting)
        return y
    if x.ndim > 3:
        raise ValueError("Input array has to have at most 3 dimensions")

    if isinstance(semiring, Reals):
        result = _issR(x, word,
            partial=partial,
            weighting=weighting,
            normalize=normalize,
        )
    elif isinstance(semiring, Arctic):
        result = _issA(x, word,
            partial=partial,
            weighting=weighting,
            indices=semiring.returns_indices,
        )
    elif isinstance(semiring, Bayesian):
        result = _issB(x, word, partial=partial, weighting=weighting)
    else:
        raise ValueError(f"Unknown semiring {semiring!r}")

    if type_ is not None:
        x = x.astype(type_)
        try:
            result = result.astype(type_)
        except:
            pass
    return result


def _issR(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    normalize: bool = False,
) -> np.ndarray:
    if weighting is None:
        if partial:
            return _partial_reals(x, word.numpy())
        return _reals(x, word.numpy(), normalize)
    elif isinstance(weighting, Exponential):
        if partial:
            if weighting.outer:
                return _partial_exp_outer_reals(
                    x,
                    word.numpy(),
                    weighting.alpha,
                    weighting.time(x.shape[0]),
                )
            return _partial_exp_reals(
                x,
                word.numpy(),
                weighting.alpha,
                weighting.time(x.shape[0]),
            )
        if weighting.outer:
            return _exp_outer_reals(
                x,
                word.numpy(),
                weighting.alpha,
                weighting.time(x.shape[0]),
            )
        return _exp_reals(
            x,
            word.numpy(),
            weighting.alpha,
            weighting.time(x.shape[0]),
        )
    elif isinstance(weighting, Cosine):
        if partial:
            raise NotImplementedError(
                "Partial cosine weighting is not implemented"
            )
        if weighting.outer:
            return _cos_outer_reals(
                x, word.numpy(),
                alpha=weighting.alpha,
                expansion=weighting.expansion(word),
                time=weighting.time(x.shape[0]),
            )
        return _cos_reals(
            x, word.numpy(),
            alpha=weighting.alpha,
            expansion=weighting.expansion(word),
            time=weighting.time(x.shape[0]),
        )
    else:
        raise NotImplementedError


def split_argmax_output(
    x: np.ndarray,
    p: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    itsum = np.empty((p, x.shape[1]))
    index = [np.empty((x.shape[1], k+1)) for k in range(p)]
    for k in range(p):
        j = int(k + (k * (k+1) / 2))
        itsum[k, :] = x[j, :]
        for c in range(k+1):
            index[k][:, c] = x[j+c+1, :]
    return itsum, index


def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    if weighting is None:
        if indices:
            array = _partial_arctic_argmax(x, word.numpy())
            itsum, index = split_argmax_output(array, len(word))
            if partial:
                return itsum, index
            return itsum[-1], [index[-1]]
        if partial:
            return _partial_arctic(x, word.numpy())
        return _arctic(x, word.numpy())
    else:
        raise NotImplementedError(
            "Weighted arctic iterated sums are not supported"
        )


def _issB(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
) -> np.ndarray:
    if np.any(x < 0):
        warnings.warn(
            "Input array contains negative numbers, which are prohibited in "
            "the Bayesian semiring. Output might not match the iterated sum.",
            RuntimeWarning,
        )
    if weighting is None:
        if partial:
            return _partial_bayesian(x, word.numpy())
        return _bayesian(x, word.numpy())
    else:
        raise NotImplementedError(
            "Weighted arctic iterated sums are not supported"
        )
