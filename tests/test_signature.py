import numpy as np

import iss


def test_base() -> None:
    x = np.random.random(size=(100, 3)) + 1
    word = iss.Word("[12^3][32][12^(-1)]")

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

    np.testing.assert_allclose(actual, iss.iss(x, word))


def test_partial() -> None:
    x = np.random.normal(size=(5, 3))
    word = iss.Word("[12^3][32][12]")

    actual = np.zeros((3, 5))

    for t in range(5):
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

    np.testing.assert_allclose(actual, iss.iss(x, word, partial=True))


def test_exp() -> None:
    x = np.random.normal(size=(100, 3))
    word = iss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 100)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 100)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 100)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Exponential(alpha),
        ),
        rtol=1e-4,
    )


def test_partial_exp() -> None:
    x = np.random.normal(size=(100, 3))
    word = iss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((3, 100))

    for t in range(100):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                * np.exp(-alpha[0] * (t - t_3) / 100)
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    * np.exp(-alpha[1] * (t - t_3) / 100)
                    * np.exp(-alpha[0] * (t_3 - t_2) / 100)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 100)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 100)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 100)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=True,
            weighting=iss.weighting.Exponential(alpha),
        ),
        rtol=1e-4,
    )
