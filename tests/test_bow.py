import numpy as np

import witss


def test_base() -> None:
    x = np.random.random((100, 2))
    words = [
        witss.Word("[1][1]"),
        witss.Word("[1][1^2][1]"),
        witss.Word("[1]"),
        witss.Word("[1][1][1][1]"),
        witss.Word("[2][1]"),
        witss.Word("[1][1][1]"),
        witss.Word("[2][1][1]"),
        witss.Word("[2]"),
    ]
    bow = witss.BagOfWords(*words)

    itsums = witss.iss(x, bow)
    for word in words:
        np.testing.assert_allclose(
            itsums[word], witss.iss(x, word)
        )
