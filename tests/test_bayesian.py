import numpy as np

import iss


def test_base() -> None:
    x = np.random.random(size=(100, 3))
    word = iss.Word("[3][21][11]")

    actual = np.zeros((100, ))

    for t in range(100):
        actual[t] = -np.inf
        for t_3 in range(t+1):
            for t_2 in range(t_3+1):
                for t_1 in range(t_2+1):
                    actual[t] = max(
                          x[t_3, 0] ** 2
                        * x[t_2, 0] * x[t_2, 1]
                        * x[t_1, 2],
                        actual[t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word, semiring=iss.semiring.Bayesian()),
    )

    actual = np.zeros((100, ))

    for t in range(100):
        actual[t] = -np.inf
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] = max(
                          x[t_3, 0] ** 2
                        * x[t_2, 0] * x[t_2, 1]
                        * x[t_1, 2],
                        actual[t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word, semiring=iss.semiring.Bayesian(), strict=True),
    )


def test_partial() -> None:
    x = np.random.random(size=(50, 3))
    word = iss.Word("[3][21][11]")

    actual = np.zeros((3, 50))

    for t in range(50):
        actual[:, t] = -np.inf
        for t_3 in range(t+1):
            actual[0, t] = max(
                x[t_3, 2],
                actual[0, t]
            )
            for t_2 in range(t_3+1):
                actual[1, t] = max(
                      x[t_2, 2]
                    * x[t_3, 0] * x[t_3, 1],
                    actual[1, t]
                )
                for t_1 in range(t_2+1):
                    actual[2, t] = max(
                          x[t_3, 0]**2
                        * x[t_2, 0] * x[t_2, 1]
                        * x[t_1, 2],
                        actual[2, t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word,
            partial=True,
            semiring=iss.semiring.Bayesian(),
        ).numpy(),
    )

    actual = np.zeros((3, 50))

    for t in range(50):
        actual[:, t] = -np.inf
        for t_3 in range(t+1):
            actual[0, t] = max(
                x[t_3, 2],
                actual[0, t]
            )
            for t_2 in range(t_3):
                actual[1, t] = max(
                      x[t_2, 2]
                    * x[t_3, 0] * x[t_3, 1],
                    actual[1, t]
                )
                for t_1 in range(t_2):
                    actual[2, t] = max(
                          x[t_3, 0]**2
                        * x[t_2, 0] * x[t_2, 1]
                        * x[t_1, 2],
                        actual[2, t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word,
            partial=True,
            semiring=iss.semiring.Bayesian(),
            strict=True,
        ).numpy(),
    )


if __name__ == "__main__":
    test_partial()
