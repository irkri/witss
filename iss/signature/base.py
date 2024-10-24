import warnings
from collections import OrderedDict
from typing import Literal, Optional, Sequence, overload

import numpy as np

from ..words.word import BagOfWords, Word
from .compute import (_arctic, _arctic_argmax, _bayesian, _cos_outer_reals,
                      _cos_reals, _exp_outer_reals, _exp_reals,
                      _partial_arctic, _partial_arctic_argmax,
                      _partial_bayesian, _partial_exp_outer_reals,
                      _partial_exp_reals, _partial_reals, _reals)
from .semiring import Arctic, Bayesian, Reals, Semiring
from .weighting import Cosine, Exponential, Weighting


class ISS:

    def __init__(
        self,
        words: list[Word],
        values: list[np.ndarray] | np.ndarray,
    ) -> None:
        self._sums = OrderedDict(zip(map(str, words), values))

    def __getitem__(self, key: Word | str) -> np.ndarray:
        return self._sums[str(Word(key) if isinstance(key, str) else key)]

    def numpy(self) -> np.ndarray:
        return np.array([i for i in self._sums.values()])

    def __repr__(self) -> str:
        return f"ISS({', '.join(self._sums.keys())})"


@overload
def iss(
    x: np.ndarray,
    word: BagOfWords | Sequence[Word] | Sequence[str],
    partial: bool = ...,
    weighting: Optional[Weighting] = ...,
    semiring: Semiring = ...,
    strict: Optional[bool] = ...,
    normalize: bool = ...,
) -> ISS:
    ...
@overload
def iss(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[False] = ...,
    weighting: Optional[Weighting] = ...,
    semiring: Semiring = ...,
    strict: Optional[bool] = ...,
    normalize: bool = ...,
) -> np.ndarray:
    ...
@overload
def iss(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[True] = ...,
    weighting: Optional[Weighting] = ...,
    semiring: Semiring = ...,
    strict: Optional[bool] = ...,
    normalize: bool = ...,
) -> ISS:
    ...
def iss(
    x: np.ndarray,
    word: BagOfWords | Word | str | Sequence[Word] | Sequence[str],
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    semiring: Semiring = Reals(),
    strict: Optional[bool] = None,
    normalize: bool = False,
) -> np.ndarray | ISS:
    """Calculate the iterated sums signature of the given time series
    evaluated at the given word.

    Args:
        x (np.ndarray): Input array of 2 dimensions ``(T, d)`` where
            ``T`` is the sequence length and ``d`` the dimension of each
            time step.
        word (BagOfWords | Word | str): The word the signature should be
            evaluated on.
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
        strict (bool, optional): Whether to use strict inequalities for
            the time steps of the iterated sum. Defaults to True for the
            Real semiring and False for every other semiring.
        normalize (bool, optional): If True, normalizes the iterated sum
            for the real semiring by normalizing each cumulative sum.
            This prevents overflow. Defaults to False.
    """
    if isinstance(word, (Word, str)):
        word = Word(word) if not isinstance(word, Word) else word
        if not partial:
            return _iss_single(
                x, word, partial, weighting, semiring, strict, normalize
            )
        array = _iss_single(
            x, word, partial, weighting, semiring, strict, normalize
        )
        return ISS(word.prefixes(), [array[i] for i in range(len(array))])

    raise NotImplementedError
    words = BagOfWords()
    for w in word:
        words = words.join(Word(w) if isinstance(w, str) else w)


def _iss_single(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    semiring: Semiring = Reals(),
    strict: Optional[bool] = None,
    normalize: bool = False,
) -> np.ndarray:
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
            y[i] = _iss_single(
                x[i],
                word=word,
                partial=partial,
                weighting=weighting,
                semiring=semiring,
                strict=strict,
                normalize=normalize,
            )
        return y
    if x.ndim > 3:
        raise ValueError("Input array has to have at most 3 dimensions")

    if isinstance(semiring, Reals):
        result = _issR(x, word,
            partial=partial,
            weighting=weighting,
            strict=True if strict is None else strict,
            normalize=normalize,
        )
    elif isinstance(semiring, Arctic):
        result = _issA(x, word,
            partial=partial,
            weighting=weighting,
            strict=False if strict is None else strict,
        )
    elif isinstance(semiring, Bayesian):
        result = _issB(x, word,
            partial=partial,
            weighting=weighting,
            strict=False if strict is None else strict,
        )
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
    strict: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    if weighting is None:
        if partial:
            return _partial_reals(x, word.numpy(), normalize, strict)
        return _reals(x, word.numpy(), normalize, strict)
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


def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: bool = False,
    strict: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Returns the arctic iterated sum and the cumulative argmax of the
    given time series. This time series can be multidimensional.

    Args:
        x (np.ndarray): Numpy array of shape ``(T,d)``, where ``d`` is
            the dimensionality of a single entry.
        word (Word | str): Can be a Word or a string representation of a
            word. For ``d=1`` dimensional time series, also a string
            consisting of ``+`` and ``-`` is allowed. Here, ``+`` means
            a maximum and ``-`` a minimum. The string ``word="+-+"``
            leads to the method returning an index array of shape
            ``(T,3)``, consisting of indices of a maximum, followed by a
            minimum, followed by a maximum.
        partial (bool, optional): Whether to compute also all prefix
            words of the supplied Word. Defaults to False.
        strict (bool, optional): Whether strict inequalities in the
            iterated sum time steps should be used. Defaults to False.

    Returns:
        tuple[np.ndarray, tuple[np.ndarray, ...]]: The first entry of
            the returned tuple is an array with the maxima. The second
            is a tuple of arrays, in which each array consists of
            indices. For ``partial=False``, this is always a single
            array of shape ``(T, len(word))``.
    """
    if isinstance(word, str) and word.replace("+", "").replace("-", "") == "":
        word = word.replace("+", "[1]").replace("-", "[1^(-1)]")
    word = Word(word) if not isinstance(word, Word) else word

    type_ = None
    if x.dtype != np.float64:
        type_ = x.dtype
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim > 2:
        raise ValueError("Input array has to have at most 2 dimensions")

    result = _issA(x, word, partial=partial, indices=True, strict=strict)

    if type_ is not None:
        x = x.astype(type_)
        try:
            result = (result[0].astype(type_), result[1])
        except:
            pass
    return result


def split_argmax_output(
    x: np.ndarray,
    p: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    itsum = x[:p, :]
    index = tuple(
        np.empty((x.shape[1], k+1), dtype=np.int32) for k in range(p)
    )
    for k in range(p):
        j = int((k * (k+1) / 2))
        for c in range(k+1):
            index[k][:, c] = x[p+j+c, :]
    return itsum, index


@overload
def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = ...,
    weighting: Optional[Weighting] = ...,
    strict: bool = ...,
    indices: Literal[False] = ...,
) -> np.ndarray:
    ...
@overload
def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = ...,
    weighting: Optional[Weighting] = ...,
    strict: bool = ...,
    indices: Literal[True] = ...,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    ...
def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    strict: bool = False,
    indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if weighting is None:
        if indices:
            if partial:
                array = _partial_arctic_argmax(x, word.numpy(), strict)
                itsum, index = split_argmax_output(array, len(word))
                return itsum, index
            array = _arctic_argmax(x, word.numpy(), strict)
            return array[0], (array[1:].astype(np.int32).swapaxes(0, 1), )
        if partial:
            return _partial_arctic(x, word.numpy(), strict)
        return _arctic(x, word.numpy(), strict)
    else:
        raise NotImplementedError(
            "Weighted arctic iterated sums are not supported"
        )


def _issB(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
    strict: bool = False,
) -> np.ndarray:
    if np.any(x < 0):
        warnings.warn(
            "Input array contains negative numbers, which are prohibited in "
            "the Bayesian semiring. Output might not match the iterated sum.",
            RuntimeWarning,
        )
    if weighting is None:
        if partial:
            return _partial_bayesian(x, word.numpy(), strict)
        return _bayesian(x, word.numpy(), strict)
    else:
        raise NotImplementedError(
            "Weighted arctic iterated sums are not supported"
        )
