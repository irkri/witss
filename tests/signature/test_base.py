import numpy as np

import witss


def test_base() -> None:
    x = np.random.random(size=(100, 3)) + 1
    word = witss.Word("[12^3][32][12^(-1)]")

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0] * x[t_3, 1] ** (-1)
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                    )

    np.testing.assert_allclose(actual, witss.iss(x, word))

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3+1):
                for t_1 in range(t_2+1):
                    actual[t] += (
                          x[t_3, 0] * x[t_3, 1] ** (-1)
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                    )

    np.testing.assert_allclose(actual, witss.iss(x, word, strict=False))

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                        x[t_3, 0] * x[t_3, 1] ** (-1)
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(actual, witss.iss(x, word, normalize=True))

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3+1):
                for t_1 in range(t_2+1):
                    actual[t] += (
                        x[t_3, 0] * x[t_3, 1] ** (-1)
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        / (t+1) / (t_3+1) / (t_2+1)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(x, word, normalize=True, strict=False),
    )


def test_partial() -> None:
    x = np.random.normal(size=(100, 3))
    word = witss.Word("[12^3][32][12]")

    actual = np.zeros((3, 100))

    for t in range(100):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                    )

    np.testing.assert_allclose(actual, witss.iss(x, word, partial=True).numpy())

    actual = np.zeros((3, 100))

    for t in range(100):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                / (t+1)
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    / (t) / (t_3)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(x, word, partial=True, normalize=True).numpy(),
    )

    actual = np.zeros((3, 100))

    for t in range(100):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
            )
            for t_2 in range(t_3+1):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                )
                for t_1 in range(t_2+1):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(x, word, partial=True, strict=False).numpy(),
    )

    actual = np.zeros((3, 100))

    for t in range(100):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                / (t+1)
            )
            for t_2 in range(t_3+1):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    / (t+1) / (t_3+1)
                )
                for t_1 in range(t_2+1):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        / (t+1) / (t_3+1) / (t_2+1)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(x, word, partial=True, strict=False, normalize=True).numpy(),
    )
