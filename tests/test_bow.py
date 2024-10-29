import numpy as np

import iss


def test_base() -> None:
    x = np.random.random((100, 2))
    words = [
        iss.Word("[1][1]"),
        iss.Word("[1][1^2][1]"),
        iss.Word("[1]"),
        iss.Word("[1][1][1][1]"),
        iss.Word("[2][1]"),
        iss.Word("[1][1][1]"),
        iss.Word("[2][1][1]"),
        iss.Word("[2]"),
    ]
    bow = iss.BagOfWords(*words)

    itsums = iss.iss(x, bow)
    for word in words:
        np.testing.assert_allclose(
            itsums[word], iss.iss(x, word)
        )
