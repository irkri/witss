import numpy as np

import witss


def test_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = witss.Word("[13][2^2][1]")
    alpha = np.array([.75, .5])

    actual = np.zeros((50, ))

    for t in range(50):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0]
                        * x[t_2, 1] **2
                        * x[t_1, 0] * x[t_1, 2]
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Exponential(alpha, outer=False),
        ),
    )

    actual = np.zeros((50, ))

    for t in range(50):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0]
                        * x[t_2, 1] **2
                        * x[t_1, 0] * x[t_1, 2]
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Exponential(alpha, outer=False),
            normalize=True,
        ),
    )


def test_outer_exp() -> None:
    x = np.random.normal(size=(2, 50, 3))
    word = witss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((2, 50))

    for n in range(2):
        for t in range(50):
            for t_3 in range(t+1):
                for t_2 in range(t_3):
                    for t_1 in range(t_2):
                        actual[n, t] += (
                            x[n, t_3, 0] * x[n, t_3, 1]
                            * x[n, t_2, 1] * x[n, t_2, 2]
                            * x[n, t_1, 0] * x[n, t_1, 1] ** 3
                            * np.exp(-alpha[2] * (t - t_3) / 50)
                            * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                            * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Exponential(alpha, outer=True),
        ),
    )

    actual = np.zeros((2, 50))

    for n in range(2):
        for t in range(50):
            for t_3 in range(t+1):
                for t_2 in range(t_3):
                    for t_1 in range(t_2):
                        actual[n, t] += (
                              x[n, t_3, 0] * x[n, t_3, 1]
                            * x[n, t_2, 1] * x[n, t_2, 2]
                            * x[n, t_1, 0] * x[n, t_1, 1] ** 3
                            * np.exp(-alpha[2] * (t - t_3) / 50)
                            * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                            * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                            / (t-1) / (t_3-1) / (t_2)
                        )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Exponential(alpha, outer=True),
            normalize=True,
        ),
    )


def test_partial_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = witss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8])

    actual = np.zeros((3, 50))

    for t in range(50):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=True,
            weighting=witss.weighting.Exponential(alpha, outer=False),
        ).numpy(),
    )

    actual = np.zeros((3, 50))

    for t in range(50):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                / (t+1)
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                    / (t) / (t_3)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=True,
            weighting=witss.weighting.Exponential(alpha, outer=False),
            normalize=True,
        ).numpy(),
    )


def test_partial_outer_exp() -> None:
    x = np.random.normal(size=(4, 50, 3))
    word = witss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((3, 4, 50))

    for n in range(4):
        for t in range(50):
            for t_3 in range(t+1):
                actual[0, n, t] += (
                    x[n, t_3, 0] * x[n, t_3, 1] ** 3
                    * np.exp(-alpha[0] * (t - t_3) / 50)
                )
                for t_2 in range(t_3):
                    actual[1, n, t] += (
                        x[n, t_2, 0] * x[n, t_2, 1] ** 3
                        * x[n, t_3, 1] * x[n, t_3, 2]
                        * np.exp(-alpha[1] * (t - t_3) / 50)
                        * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                    )
                    for t_1 in range(t_2):
                        actual[2, n, t] += (
                              x[n, t_3, 0] * x[n, t_3, 1]
                            * x[n, t_2, 1] * x[n, t_2, 2]
                            * x[n, t_1, 0] * x[n, t_1, 1] ** 3
                            * np.exp(-alpha[2] * (t - t_3) / 50)
                            * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                            * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            batches=1,
            partial=True,
            weighting=witss.weighting.Exponential(alpha),
        ).numpy(),
    )

    actual = np.zeros((3, 4, 50))

    for n in range(4):
        for t in range(50):
            for t_3 in range(t+1):
                actual[0, n, t] += (
                    x[n, t_3, 0] * x[n, t_3, 1] ** 3
                    * np.exp(-alpha[0] * (t - t_3) / 50)
                    / (t+1)
                )
                for t_2 in range(t_3):
                    actual[1, n, t] += (
                        x[n, t_2, 0] * x[n, t_2, 1] ** 3
                        * x[n, t_3, 1] * x[n, t_3, 2]
                        * np.exp(-alpha[1] * (t - t_3) / 50)
                        * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                        / (t) / (t_3)
                    )
                    for t_1 in range(t_2):
                        actual[2, n, t] += (
                            x[n, t_3, 0] * x[n, t_3, 1]
                            * x[n, t_2, 1] * x[n, t_2, 2]
                            * x[n, t_1, 0] * x[n, t_1, 1] ** 3
                            * np.exp(-alpha[2] * (t - t_3) / 50)
                            * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                            * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                            / (t-1) / (t_3-1) / (t_2)
                        )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            batches=2,
            partial=True,
            weighting=witss.weighting.Exponential(alpha),
            normalize=True,
        ).numpy(),
    )
