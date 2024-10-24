import re
from typing import Optional, Self

import numpy as np


class Word:
    """Words evaluate the iterated sums signature at a certain level."""

    RE = re.compile(r"(\[(((-?\d)|\((-?\d+)\))(\^((-?\d)|\((-?\d+)\)))?)+\])+")
    RE_SINGLE = re.compile(r"(?:(\d)|\((\d+)\))(?:\^(?:(-?\d)|\((-?\d+)\)))?")

    def __init__(self, word: Optional[str] = None) -> None:
        self._letters: list[list[tuple[int, int]]] = []
        if word is not None:
            self.multiply(word)

    @property
    def max_dim(self) -> int:
        return max(l[0] for el in self._letters for l in el)

    def is_empty(self) -> bool:
        return len(self._letters) == 0

    def multiply(self, word: str | Self | list[list[tuple[int, int]]]) -> None:
        if isinstance(word, str):
            if word == "":
                return
            if Word.RE.fullmatch(word) is None:
                raise ValueError("Input string has invalid format")
            for bracket in word.split("]")[:-1]:
                self._letters.append([])
                bracket = bracket[1:]
                dimexps = [
                    ("".join(x[:2]), "".join(x[2:]))
                    for x in Word.RE_SINGLE.findall(bracket)
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
        elif isinstance(word, Word):
            for el in word._letters:
                self._letters.append(el.copy())
        elif isinstance(word, list):
            for el in word:
                self._letters.append(el.copy())
        else:
            raise NotImplementedError

    def numpy(self) -> np.ndarray:
        exps = np.zeros((len(self._letters), self.max_dim), dtype=np.int32)
        for iel, el in enumerate(self._letters):
            for l in el:
                exps[iel, l[0]-1] = l[1]
        return exps

    def deconcat(self) -> list[tuple[Self, Self]]:
        """Deconcatenates the word into all possiible word pairs that
        form this word when multiplied together.

        Returns:
            list[tuple[Word, Word]]: _description_
        """
        pairs = [(Word(), self.copy())]
        for i in range(1, len(self._letters)+1):
            pairs.append(
                (Word() * self._letters[:i], Word() * self._letters[i:])
            )
        return pairs

    def prefixes(self) -> list[Self]:
        """Returns all prefixes of this Word, including the Word itself.
        They are ordered by length, from smallest to largest prefix.
        """
        prefixes = []
        for i in range(len(self._letters)):
            prefixes.append(Word() * self._letters[:i+1])
        return prefixes

    def copy(self) -> Self:
        return Word() * self

    def __mul__(self, word: str | Self | list[list[tuple[int, int]]] ) -> Self:
        new_word = Word()
        new_word.multiply(self)
        new_word.multiply(word)
        return new_word

    def __rmul__(self, word: str | list[list[tuple[int, int]]]) -> Self:
        new_word = Word()
        new_word.multiply(word)
        new_word.multiply(self)
        return new_word

    def __eq__(self, word: Self) -> bool:
        if not isinstance(word, (Word, str, list)):
            raise NotImplementedError(
                f"Cannot compare Word to object of type {type(word)!r}"
            )
        if not isinstance(word, Word):
            word = Word(word)
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
    """A bag of words contains a collection of Word class instances. It
    is used to speed up calculation of iterated sums evaluated on a lot
    of words in which some prefix words may overlap.
    """

    def __init__(self, *words: Word) -> None:
        self._words = [*words]

    def join(self, other: Self | Word) -> Self:
        if isinstance(other, Word):
            return BagOfWords(*self._words, other)
        return BagOfWords(*self._words, *other._words)
