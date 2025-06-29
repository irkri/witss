import re
from typing import Self

import numpy as np


class Word:
    """Words evaluate the iterated sums signature at a certain level.
    The easiest way to initialize a word is to supply a string of the
    form ``"[i_1^k_1...i_a^k_a]...[j_1^l_1...j_b^l_b]"``, where ``i_*``
    and ``j_*`` are dimensions in a time series (indexing starts at 1),
    and ``k_*`` and ``l_*`` are optional exponents.
    For two-digit dimension indices or exponents and negative exponents,
    brackets are required.

    Example:
    ```
        Word("[112^4][3^(-2)][(10)^4]")
    ```
    """

    RE = re.compile(r"(\[(((-?\d)|\((-?\d+)\))(\^((-?\d)|\((-?\d+)\)))?)+\])+")
    RE_SINGLE = re.compile(r"(?:(\d)|\((\d+)\))(?:\^(?:(-?\d)|\((-?\d+)\)))?")

    def __init__(
        self,
        word: str | Self | list[list[tuple[int, int]]] | None = None,
    ) -> None:
        self._letters: list[list[tuple[int, int]]] = []
        if word is not None:
            self.multiply(word)

    @property
    def max_dim(self) -> int:
        """The maximum dimension indices (letters) in the word."""
        return max(l[0] for el in self._letters for l in el)

    def is_empty(self) -> bool:
        return len(self._letters) == 0

    def multiply(self, word: str | Self | list[list[tuple[int, int]]]) -> None:
        """Multiplies two words together. They can be given as strings,
        :class:`Word` instances or lists of lists of integer pairs.
        The multiplication is not commutative and the ``*`` operator can
        be used as an alternative.
        """
        if isinstance(word, str):
            if word == "":
                return
            if self.__class__.RE.fullmatch(word) is None:
                raise ValueError("Input string has invalid format")
            for bracket in word.split("]")[:-1]:
                self._letters.append([])
                bracket = bracket[1:]
                dimexps = [
                    ("".join(x[:2]), "".join(x[2:]))
                    for x in self.__class__.RE_SINGLE.findall(bracket)
                ]
                dimexps = [
                    (int(dim), int(exp) if exp != "" else 1)
                    for dim, exp in dimexps
                ]
                collected_dim = []
                collected_exp = []
                for dimexp in dimexps:
                    if dimexp[0] in collected_dim:
                        index = collected_dim.index(dimexp[0])
                        collected_exp[index] += dimexp[1]
                    else:
                        collected_dim.append(dimexp[0])
                        collected_exp.append(dimexp[1])
                while 0 in collected_exp:
                    ind = collected_exp.index(0)
                    collected_dim.pop(ind)
                    collected_exp.pop(ind)
                self._letters[-1].extend(zip(collected_dim, collected_exp))
        elif isinstance(word, self.__class__):
            for el in word._letters:
                self._letters.append(el.copy())
        elif isinstance(word, list):
            for el in word:
                self._letters.append(el.copy())
        else:
            raise NotImplementedError

    def numpy(self) -> np.ndarray:
        """Returns a numpy array representation of the word."""
        exps = np.zeros((len(self._letters), self.max_dim), dtype=np.int32)
        for iel, el in enumerate(self._letters):
            for l in el:
                exps[iel, l[0]-1] = l[1]
        return exps

    def deconcat(self) -> list[tuple[Self, Self]]:
        """Deconcatenates the word into all possiible word pairs that
        form this word when multiplied together.
        """
        pairs = [(self.__class__(), self.copy())]
        for i in range(1, len(self._letters)+1):
            pairs.append((
                self.__class__() * self._letters[:i],
                self.__class__() * self._letters[i:],
            ))
        return pairs

    def prefixes(self) -> list[Self]:
        """Returns all prefixes of this Word, including the Word itself.
        They are ordered by length, from smallest to largest prefix.
        """
        prefixes = []
        for i in range(len(self._letters)):
            prefixes.append(self.__class__() * self._letters[:i+1])
        return prefixes

    def copy(self) -> Self:
        """Returns a copy of this word."""
        return self.__class__() * self

    def __mul__(self, word: str | Self | list[list[tuple[int, int]]] ) -> Self:
        new_word = self.__class__()
        new_word.multiply(self)
        new_word.multiply(word)
        return new_word

    def __rmul__(self, word: str | list[list[tuple[int, int]]]) -> Self:
        new_word = self.__class__()
        new_word.multiply(word)
        new_word.multiply(self)
        return new_word

    def __contains__(self, other: Self) -> bool:
        for i in range(len(self)):
            if other._letters == self._letters[:i]:
                return True
        return False

    def __eq__(self, word: Self) -> bool:
        if not isinstance(word, (self.__class__, str, list)):
            raise NotImplementedError(
                f"Cannot compare Word to object of type {type(word)!r}"
            )
        if not isinstance(word, self.__class__):
            word = self.__class__(word)
        if len(word) != len(self):
            return False

        for k in range(len(word)):
            dims, exps = zip(*self._letters[k])
            for dim, exp in word._letters[k]:
                if dim not in dims:
                    return False
                if exps[dims.index(dim)] != exp:
                    return False
        return True

    def __len__(self) -> int:
        return len(self._letters)

    def __str__(self) -> str:
        strings = []
        for dimexps in self._letters:
            string = ""
            for dimexp in dimexps:
                string += f"({dimexp[0]})" if dimexp[0] > 9 else str(dimexp[0])
                if dimexp[1] != 1:
                    string += "^"
                    string += (
                        f"({dimexp[1]})" if (dimexp[1] > 9 or dimexp[1] < 0)
                        else str(dimexp[1])
                    )
            strings.append(string)
        return (
            "[" + "][".join(strings) + "]"
        )

    def __repr__(self) -> str:
        return f"Word({str(self)})"


class BagOfWords:
    """A BagOfWords contains a collection of Word class instances. It
    is used to speed up calculation of iterated sums evaluated on a lot
    of words in which some prefix words may overlap. Processing all
    words is done once at initialization of a BagOfWords and takes time
    proportional to the number of words inside squared.
    """

    def __init__(self, *words: Word | str) -> None:
        self._words = [
            word if isinstance(word, Word) else Word(word)
            for word in words
        ]
        self._process()

    def _process(self) -> None:
        self._words = sorted(self._words, key=lambda x: len(x), reverse=True)
        words_r = self._words[::-1]
        references: list[None | tuple[int, int]] = []
        partial_flag = [False for _ in range(len(self._words))]
        for word in words_r:
            for j, larger_word in enumerate(self._words):
                if word in larger_word and word != larger_word:
                    references.append((j, len(word)))
                    partial_flag[j] = True
                    break
            else:
                references.append(None)
        self._references = references[::-1]
        self._partial_flags = partial_flag

    def join(self, other: Self | Word) -> Self:
        """Joins this BagOfWords with another and returns the resulting
        instance.
        """
        if isinstance(other, Word):
            return self.__class__(*self._words, other)
        return self.__class__(*self._words, *other._words)

    def words(self) -> list[Word]:
        """Returns a list of words in this BagOfWords."""
        return self._words

    def __getitem__(self, index: int) -> tuple[Word, bool | tuple[int, int]]:
        ref = self._references[index]
        return (
            self._words[index],
            ref if ref is not None
                else self._partial_flags[index],
        )

    def __len__(self) -> int:
        return len(self._words)

    def __iter__(self) -> Self:
        self._i = -1
        return self

    def __next__(self) -> tuple[Word, bool | tuple[int, int]]:
        self._i += 1
        if self._i == len(self):
            raise StopIteration
        ref = self._references[self._i]
        return (
            self._words[self._i],
            ref if ref is not None
                else self._partial_flags[self._i],
        )
