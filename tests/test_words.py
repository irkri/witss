import numpy as np

import iss

def test_repr() -> None:
    assert str(iss.Word("[1(10)^21^43][243^953]")) == "[1^5(10)^23][243^(10)5]"

    assert str(iss.Word("[1^(-10)3^(-3)3^213]")) == "[1^(-9)]"


def test_operations() -> None:
    word1 = iss.Word("[1^3][3^4]")
    word2 = iss.Word("[2^14^(-1)][(10)]")
    word = word1*word2

    assert str(word) == str(word1*"[2^14^(-1)][(10)]")
    assert str(word) == str(word1*[[(2, 1), (4, -1)], [(10, 1)]])
    assert str(word) == "[1^3][3^4][24^(-1)][(10)]"

    np.testing.assert_equal(
        np.array([
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]),
        word.numpy(),
    )

    assert word == iss.Word("[111][3^233][4^(3)24^(-4)][2^1(10)2^(-1)]")

    assert word.deconcat() == [
        (iss.Word(), iss.Word("[1^3][3^4][24^(-1)][(10)]")),
        (iss.Word("[1^3]"), iss.Word("[3^4][24^(-1)][(10)]")),
        (iss.Word("[1^3][3^4]"), iss.Word("[24^(-1)][(10)]")),
        (iss.Word("[1^3][3^4][24^(-1)]"), iss.Word("[(10)]")),
        (iss.Word("[1^3][3^4][24^(-1)][(10)]"), iss.Word()),
    ]

