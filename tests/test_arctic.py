import numpy as np

import iss


def test_base() -> None:
    x = np.random.random(size=(100, 3))
    word = iss.Word("[3^(-1)][21][11]")

    actual = np.zeros((100, ))

    for t in range(100):
        actual[t] = -np.inf
        for t_3 in range(t+1):
            for t_2 in range(t_3+1):
                for t_1 in range(t_2+1):
                    actual[t] = max(
                        2 * x[t_3, 0]
                        + x[t_2, 0] + x[t_2, 1]
                        - x[t_1, 2],
                        actual[t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word, semiring=iss.semiring.Arctic()),
    )

    actual = np.zeros((100, ))

    for t in range(100):
        actual[t] = -np.inf
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] = max(
                        2 * x[t_3, 0]
                        + x[t_2, 0] + x[t_2, 1]
                        - x[t_1, 2],
                        actual[t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word, semiring=iss.semiring.Arctic(), strict=True),
    )



def test_partial() -> None:
    x = np.random.normal(size=(100, 3))
    word = iss.Word("[3^(-1)][21][11]")

    actual = np.zeros((3, 100))

    for t in range(100):
        actual[:, t] = -np.inf
        for t_3 in range(t+1):
            actual[0, t] = max(
                - x[t_3, 2],
                actual[0, t]
            )
            for t_2 in range(t_3+1):
                actual[1, t] = max(
                    - x[t_2, 2]
                    + x[t_3, 0] + x[t_3, 1],
                    actual[1, t]
                )
                for t_1 in range(t_2+1):
                    actual[2, t] = max(
                          2 * x[t_3, 0]
                        + x[t_2, 0] + x[t_2, 1]
                        - x[t_1, 2],
                        actual[2, t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word, partial=True, semiring=iss.semiring.Arctic()).numpy(),
    )

    actual = np.zeros((3, 100))

    for t in range(100):
        actual[:, t] = -np.inf
        for t_3 in range(t+1):
            actual[0, t] = max(
                - x[t_3, 2],
                actual[0, t]
            )
            for t_2 in range(t_3):
                actual[1, t] = max(
                    - x[t_2, 2]
                    + x[t_3, 0] + x[t_3, 1],
                    actual[1, t]
                )
                for t_1 in range(t_2):
                    actual[2, t] = max(
                          2 * x[t_3, 0]
                        + x[t_2, 0] + x[t_2, 1]
                        - x[t_1, 2],
                        actual[2, t]
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(x, word,
            partial=True,
            semiring=iss.semiring.Arctic(),
            strict=True,
        ).numpy(),
    )


def test_argmax() -> None:
    x = np.array([
        [1, 0, -4, 1, -4, 5],
        [-4, -7, -3, -2, -5, -8],
        [1, 2, 6, 2, 9, -2],
    ]).swapaxes(0, 1).astype(np.float64)

    array, indices = iss.cumargmax(x, iss.Word("[1][2][3]"), partial=True)
    np.testing.assert_allclose(
        array,
        np.array([
            [1, 1, 1, 1, 1, 5],
            [-3, -3, -2, -1, -1, -1],
            [-2, -1, 4, 4, 8, 8],
        ])
    )
    np.testing.assert_allclose(
        indices[0],
        np.array([[0], [0], [0], [3], [3], [5]])
    )
    np.testing.assert_allclose(
        indices[1],
        np.array([
            [0., 0.], [0., 0.], [0., 2.], [3., 3.], [3., 3.], [3., 3.],
        ])
    )
    np.testing.assert_allclose(
        indices[2],
        np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 2., 2.],
            [0., 2., 2.],
            [3., 3., 4.],
            [3., 3., 4.],
        ])
    )

    array, indices = iss.cumargmax(x, iss.Word("[1][1]"), strict=True)
    np.testing.assert_allclose(
        array[1:],
        np.array([1., 1., 2., 2., 6.])
    )
    np.testing.assert_allclose(
        indices[0],
        np.array([[0, 0], [0, 1], [0, 1], [0, 3], [0, 3], [3, 5]])
    )


if __name__ == "__main__":
    test_argmax()
