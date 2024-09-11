import re
from typing import Optional, Self

import numpy as np


class Word:
    """Words evaluate the iterated sums signature at a certain level."""

    FORMAT = re.compile(r"(?:(\d)|\((\d+)\))(?:\^(?:(-?\d)|\((-?\d+)\)))?")

    def __init__(self, word: Optional[str] = None) -> None:
        self._letters: list[list[tuple[int, int]]] = []
        if word is not None:
            self.multiply(word)

    def multiply(self, word: str | Self) -> None:
        if isinstance(word, str):
            for bracket in word.split("]")[:-1]:
                self._letters.append([])
                bracket = bracket[1:]
                dimexps = [
                    ("".join(x[:2]), "".join(x[2:]))
                    for x in Word.FORMAT.findall(bracket)
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
                self._letters[-1].extend(zip(collected_dim, collected_exp))
        elif isinstance(word, Word):
            self._letters.extend(word._letters)

    def __mul__(self, word: str | Self) -> Self:
        new_word = Word()
        new_word.multiply(self)
        new_word.multiply(word)
        return new_word

    def __rmul__(self, word: str) -> Self:
        new_word = Word()
        new_word.multiply(word)
        new_word.multiply(self)
        return new_word

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([[a[0] for a in l] for l in self._letters]),
            np.array([[a[1] for a in l] for l in self._letters]),
        )

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
