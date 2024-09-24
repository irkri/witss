import itertools
from typing import Generator

from .word import Word


def _partitions_of(
    n: int,
    start: int = 1,
) -> Generator[tuple[int, ...], None, None]:
    yield (n,)
    for i in range(start, n//2 + 1):
        for p in _partitions_of(n-i, i):
            yield (i,) + p


def _extended_letters_by_weight(w: int, d: int = 1) -> list[str]:
    return [
        "["
        + "".join(["(" + str(x) + ")" if len(str(x)) > 1 else str(x)
                   for x in el])
        + "]"
        for el in
        itertools.combinations_with_replacement(list(range(1, d+1)), w)
    ]


def of_weight(w: int, dim: int = 1) -> tuple[Word, ...]:
    """Returns a list of all words that have exactly the given number of
    letters ('weight' of the words). For ``w=2`` and ``dim=2``, returns
    a list containing::

        Word("[11]"), Word("[12]"), Word("[22]"), Word("[1][1]"),
        Word("[1][2]"), Word("[2][1]"), Word("[2][2]")

    Args:
        w (int): Weight of the words, i.e. number of letters.
        dim (int, optional): Highest dimension of a letter used.
            Defaults to 1.
    """
    extended_letters = []
    words = []
    for i in range(1, w+1):
        extended_letters.append(_extended_letters_by_weight(i, dim))
    for partition in _partitions_of(w):
        for mixed_up_partition in set(itertools.permutations(partition)):
            raw_words = itertools.product(*[extended_letters[weight-1]
                                            for weight in mixed_up_partition])
            for raw_word in raw_words:
                words.append(Word("".join(raw_word)))
    return tuple(words)
