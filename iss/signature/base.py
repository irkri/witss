from typing import Optional

import numpy as np

from ..words.word import Word
from .compute import (_cos_iterated_sums_compiled,
                      _cos_outer_iterated_sums_compiled,
                      _exp_iterated_sums_compiled,
                      _exp_outer_iterated_sums_compiled,
                      _iterated_sums_compiled,
                      _partial_exp_iterated_sums_compiled,
                      _partial_exp_outer_iterated_sums_compiled,
                      _partial_iterated_sums_compiled)
from .weighting import Cosine, Exponential, Weighting


def iss(
    x: np.ndarray,
    word: Word | str,
    partial: bool = False,
    weighting: Optional[Weighting] = None,
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
    """
    word = word if isinstance(word, Word) else Word(word)
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
    if weighting is None:
        if partial:
            return _partial_iterated_sums_compiled(x, word.numpy())
        return _iterated_sums_compiled(x, word.numpy())
    elif isinstance(weighting, Exponential):
        if partial:
            if weighting.outer:
                return _partial_exp_outer_iterated_sums_compiled(
                    x,
                    word.numpy(),
                    weighting.alpha,
                    weighting.time(x.shape[0]),
                )
            return _partial_exp_iterated_sums_compiled(
                x,
                word.numpy(),
                weighting.alpha,
                weighting.time(x.shape[0]),
            )
        if weighting.outer:
            return _exp_outer_iterated_sums_compiled(
                x,
                word.numpy(),
                weighting.alpha,
                weighting.time(x.shape[0]),
            )
        return _exp_iterated_sums_compiled(
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
            return _cos_outer_iterated_sums_compiled(
                x, word.numpy(),
                alpha=weighting.alpha,
                expansion=weighting.expansion(word),
                time=weighting.time(x.shape[0]),
            )
        return _cos_iterated_sums_compiled(
            x, word.numpy(),
            alpha=weighting.alpha,
            expansion=weighting.expansion(word),
            time=weighting.time(x.shape[0]),
        )
    else:
        raise NotImplementedError
