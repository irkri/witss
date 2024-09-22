import numpy as np

from ..words.word import Word
from .compute import _iterated_sums_compiled, _partial_iterated_sums_compiled


def iss(
    x: np.ndarray,
    word: Word | str,
    partial: bool = False,
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
            for ``word=[1]`` and ``word=[1][2]``.
    """
    word = word if isinstance(word, Word) else Word(word)
    if not partial:
        return _iterated_sums_compiled(x, word.numpy())
    else:
        return _partial_iterated_sums_compiled(x, word.numpy())
