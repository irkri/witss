import warnings
from collections import OrderedDict
from math import ceil
from typing import Literal, Sequence, overload

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
    word: BagOfWords | Sequence[Word | str],
    batches: int = 1,
    partial: Literal[False] = ...,
    weighting: Weighting | None = ...,
    semiring: Semiring | None = ...,
    strict: bool | None = ...,
    normalize: Literal["sqrt", "linear"] | None = ...,
) -> ISS:
    ...
@overload
def iss(
    x: np.ndarray,
    word: Word | str,
    batches: int = 1,
    partial: Literal[False] = ...,
    weighting: Weighting | None = ...,
    semiring: Semiring | None = ...,
    strict: bool | None = ...,
    normalize: Literal["sqrt", "linear"] | None = ...,
) -> np.ndarray:
    ...
@overload
def iss(
    x: np.ndarray,
    word: Word | str,
    batches: int = 1,
    partial: Literal[True] = ...,
    weighting: Weighting | None = ...,
    semiring: Semiring | None = ...,
    strict: bool | None = ...,
    normalize: Literal["sqrt", "linear"] | None = ...,
) -> ISS:
    ...
def iss(
    x: np.ndarray,
    word: BagOfWords | Word | str | Sequence[Word | str],
    batches: int = 1,
    partial: bool = False,
    weighting: Weighting | None = None,
    semiring: Semiring | None = None,
    strict: bool | None = None,
    normalize: Literal["sqrt", "linear"] | None = None,
) -> np.ndarray | ISS:
    """Calculate the iterated sums signature of the given time series
    evaluated at the given word.

    Args:
        x (np.ndarray): Input array of at most 3 dimensions
            ``(N, T, D)`` where ``N`` is the number of given time
            series, ``T`` is the sequence length and ``D`` is the
            dimension of each time step. A 2D array is interpreted as
            having shape ``(T, D)`` and a 1D array having shape
            ``(T, )``.
        word (BagOfWords | Word | str | Sequence of Word or str): The
            word the signature should be evaluated on.
        batches (int, optional): Computes the input in batches of the
            given size. Time series in one batch are processed in
            parallel. Only works if the input time series has three
            dimensions. Defaults to 1.
        partial (bool, optional): If True, also evaluates the signature
            for all prefix words of the given word, e.g. if
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
        normalize (str, optional): This is a convenience argument that
            typically is defined in the Reals semiring. It only gets
            processed if a semiring is not specified. Defaults to None.

    Returns:
        np.ndarray | ISS: Either a single array of iterated sums
            evaluated for one word or an :class:`ISS` object that maps a
            word to one or more numpy arrays. Each numpy array is of
            shape ``(N, T)`` or ``(T, )``, based on the shape of the
            input.
    """
    if semiring is None:
        if normalize is not None:
            semiring = Reals(normalize)
        else:
            semiring = Reals()
    if isinstance(word, (Word, str)):
        word = Word(word) if not isinstance(word, Word) else word
        if not partial:
            return _iss_single(
                x, word,
                batches=batches,
                partial=partial,
                weighting=weighting,
                semiring=semiring,
                strict=strict,
            )
        array = _iss_single(
            x, word,
            batches=batches,
            partial=partial,
            weighting=weighting,
            semiring=semiring,
            strict=strict,
        )
        return ISS(word.prefixes(), [array[i] for i in range(len(array))])

    if not isinstance(word, BagOfWords):
        word = BagOfWords(*word)
    itsums = []
    for w in word:
        if isinstance(w[1], tuple):
            itsums.append(itsums[w[1][0]][w[0]].copy())
        else:
            itsums.append(
                iss(x, w[0],
                    batches=batches,
                    partial=w[1],
                    weighting=weighting,
                    semiring=semiring,
                    strict=strict,
                )
            )
    for i in range(len(itsums)):
        if isinstance(itsums[i], ISS):
            itsums[i] = itsums[i][word[i][0]]
    return ISS(word.words(), itsums)


def _iss_single(
    x: np.ndarray,
    word: Word,
    batches: int = 1,
    partial: bool = False,
    weighting: Weighting | None = None,
    semiring: Semiring = Reals(),
    strict: bool | None = None,
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    type_ = None
    if x.dtype != np.float64:
        type_ = x.dtype
        x = x.astype(np.float64)

    if word.is_empty():
        return np.ones((x.shape[0], ), dtype=np.float64)
    orig_dim = x.ndim
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
    if x.ndim == 3 and batches < x.shape[0]:
        if partial:
            y = np.zeros((len(word), x.shape[0], x.shape[1]))
        else:
            y = np.zeros((x.shape[0], x.shape[1]))
        for n in range(ceil(x.shape[0] / batches)):
            y_n = _iss_single(
                x[n*batches:(n+1)*batches],
                word=word,
                batches=batches,
                partial=partial,
                weighting=weighting,
                semiring=semiring,
                strict=strict,
            )
            if partial:
                y[:, n*batches:(n+1)*batches] = y_n
            else:
                y[n*batches:(n+1)*batches] = y_n
        return y
    if x.ndim > 3:
        raise ValueError("Input array has to have at most 3 dimensions")

    if isinstance(semiring, Reals):
        result = _issR(x, word,
            partial=partial,
            weighting=weighting,
            strict=True if strict is None else strict,
            normalization=semiring.normalization(x.shape[1]),
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
    if orig_dim < 3:
        if partial:
            result = result[:, 0]
        else:
            result = result[0]
    return result


def _issR(
    x: np.ndarray,
    word: Word,
    partial: bool,
    weighting: Weighting | None,
    strict: bool,
    normalization: np.ndarray,
) -> np.ndarray:
    if weighting is None:
        if partial:
            return _partial_reals(x, word.numpy(), normalization, strict)
        return _reals(x, word.numpy(), normalization, strict)
    elif isinstance(weighting, Exponential):
        if partial:
            if weighting.outer:
                return _partial_exp_outer_reals(
                    x, word.numpy(),
                    alpha=weighting.alpha,
                    time=weighting.time(x.shape[1]),
                    div=normalization,
                )
            return _partial_exp_reals(
                x, word.numpy(),
                alpha=weighting.alpha,
                time=weighting.time(x.shape[1]),
                div=normalization,
            )
        if weighting.outer:
            return _exp_outer_reals(
                x, word.numpy(),
                alpha=weighting.alpha,
                time=weighting.time(x.shape[1]),
                div=normalization,
            )
        return _exp_reals(
            x, word.numpy(),
            alpha=weighting.alpha,
            time=weighting.time(x.shape[1]),
            div=normalization,
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
                time=weighting.time(x.shape[1]),
                div=normalization,
            )
        return _cos_reals(
            x, word.numpy(),
            alpha=weighting.alpha,
            expansion=weighting.expansion(word),
            time=weighting.time(x.shape[1]),
            div=normalization,
        )
    else:
        raise NotImplementedError


@overload
def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[False] = ...,
    strict: bool = ...,
    return_iss: Literal[False] = ...,
) -> np.ndarray:
    ...
@overload
def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[False] = ...,
    strict: bool = ...,
    return_iss: Literal[True] = ...,
) -> tuple[np.ndarray, np.ndarray]:
    ...
@overload
def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[True] = ...,
    strict: bool = ...,
    return_iss: Literal[False] = ...,
) -> tuple[np.ndarray, ...]:
    ...
@overload
def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: Literal[True] = ...,
    strict: bool = ...,
    return_iss: Literal[True] = ...,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    ...
def cumargmax(
    x: np.ndarray,
    word: Word | str,
    partial: bool = False,
    strict: bool = False,
    return_iss: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, ...]
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, tuple[np.ndarray, ...]]
):
    """Returns the cumulative argmaxima or argminima of the given time
    series. This time series can be multidimensional.
    For example, `cumargmax(x, "++", strict=True)` returns the indices
    of the two largest values in `x`.

    Args:
        x (np.ndarray): Numpy array of shape ``(T,d)``, where ``d`` is
            the dimensionality of a single entry.
        word (Word | str): Can be a Word or a string representation of a
            word. For ``d=1`` dimensional time series, also a string
            consisting of ``+`` and ``-`` is allowed. Here, ``+`` means
            a maximum and ``-`` a minimum. The string ``word="+-+"``
            returns an index array of shape ``(T,3)``, where
            ``result[t, 0:3]`` are indices of an optimal combination of
            max-min-max in the input array up to time step ``t``.
        partial (bool, optional): Whether to compute also all prefix
            words of the supplied Word. This leads to more than one
            array being returned. Defaults to False.
        strict (bool, optional): Whether strict inequalities in the
            iterated sum time steps should be used. Defaults to False.
        return_iss (bool, optional): If set to True, also returns the
            arctic iterated sums (cumulative maxima/minima). Defaults
            to False.

    Returns:
        np.ndarray | tuple[np.ndarray, ...]: A tuple of arrays, in which
            each array consists of indices. For ``partial=False``, this
            is always a single array of shape ``(T, len(word))``.
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
    if not partial:
        return (result[0], result[1][0]) if return_iss else result[1][0]
    return result if return_iss else result[1]


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
    weighting: Weighting | None = ...,
    strict: bool = ...,
    indices: Literal[False] = ...,
) -> np.ndarray:
    ...
@overload
def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = ...,
    weighting: Weighting | None = ...,
    strict: bool = ...,
    indices: Literal[True] = ...,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    ...
def _issA(
    x: np.ndarray,
    word: Word,
    partial: bool = False,
    weighting: Weighting | None = None,
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
    weighting: Weighting | None = None,
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
