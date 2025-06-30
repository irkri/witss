import numpy as np

import witss


def test_repr() -> None:
    assert (
        str(witss.Word("[1(10)^21^43][243^953]")) == "[1^5(10)^23][243^(10)5]"
    )

    assert str(witss.Word("[1^(-10)3^(-3)3^213]")) == "[1^(-9)]"


def test_operations() -> None:
    word1 = witss.Word("[1^3][3^4]")
    word2 = witss.Word("[2^14^(-1)][(10)]")
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

    assert word == witss.Word("[111][3^233][4^(3)24^(-4)][2^1(10)2^(-1)]")

    assert word.deconcat() == [
        (witss.Word(), witss.Word("[1^3][3^4][24^(-1)][(10)]")),
        (witss.Word("[1^3]"), witss.Word("[3^4][24^(-1)][(10)]")),
        (witss.Word("[1^3][3^4]"), witss.Word("[24^(-1)][(10)]")),
        (witss.Word("[1^3][3^4][24^(-1)]"), witss.Word("[(10)]")),
        (witss.Word("[1^3][3^4][24^(-1)][(10)]"), witss.Word()),
    ]
