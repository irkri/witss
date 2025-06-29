import numpy as np

import witss


def test_cos() -> None:
    x = np.random.random((50, 3))
    word = witss.Word("[12][2][33]")
    alpha = np.pi * np.array([-.2, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 0] * x[t_1, 1]
                        * x[t_2, 1]
                        * x[t_3, 2] ** 2
                        * np.cos(alpha[1] * (t_3 - t_2) / 50)
                        * np.cos(alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=1, outer=False),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = witss.Word("[2^2][1][13^3]")
    alpha = np.pi * np.array([1, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 1] ** 2
                        * x[t_2, 0]
                        * x[t_3, 0] * x[t_3, 2] ** 3
                        * np.cos(alpha[1] * (t_3 - t_2) / 50) ** 2
                        * np.cos(alpha[0] * (t_2 - t_1) / 50) ** 2
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=2, outer=False),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = witss.Word("[12][2][33]")
    alpha = np.pi * np.array([-.2, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 0] * x[t_1, 1]
                        * x[t_2, 1]
                        * x[t_3, 2] ** 2
                        * np.cos(alpha[1] * (t_3 - t_2) / 50)
                        * np.cos(alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=1, outer=False),
            normalize=True,
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = witss.Word("[2^2][1][13^3]")
    alpha = np.pi * np.array([1, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 1] ** 2
                        * x[t_2, 0]
                        * x[t_3, 0] * x[t_3, 2] ** 3
                        * np.cos(alpha[1] * (t_3 - t_2) / 50) ** 2
                        * np.cos(alpha[0] * (t_2 - t_1) / 50) ** 2
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=2, outer=False),
            normalize=True,
        ),
        rtol=1e-4,
    )

def test_cos_outer() -> None:
    x = np.random.random((50, 3))
    word = witss.Word("[12][2][33]")
    alpha = np.pi * np.array([.4, .8, 2])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 0] * x[t_1, 1]
                        * x[t_2, 1]
                        * x[t_3, 2] ** 2
                        * np.cos(alpha[2] * (t - t_3) / 50)
                        * np.cos(alpha[1] * (t_3 - t_2) / 50)
                        * np.cos(alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=1, outer=True),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = witss.Word("[2^2][1][13^3]")
    alpha = np.pi * np.array([1.5, -.32, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 1] ** 2
                        * x[t_2, 0]
                        * x[t_3, 0] * x[t_3, 2] ** 3
                        * np.cos(alpha[2] * (t - t_3) / 50) ** 2
                        * np.cos(alpha[1] * (t_3 - t_2) / 50) ** 2
                        * np.cos(alpha[0] * (t_2 - t_1) / 50) ** 2
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=2, outer=True),
        ),
        rtol=1e-4,
    )


    x = np.random.random((50, 3))
    word = witss.Word("[12][2][33]")
    alpha = np.pi * np.array([.4, .8, 2])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 0] * x[t_1, 1]
                        * x[t_2, 1]
                        * x[t_3, 2] ** 2
                        * np.cos(alpha[2] * (t - t_3) / 50)
                        * np.cos(alpha[1] * (t_3 - t_2) / 50)
                        * np.cos(alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=1, outer=True),
            normalize=True,
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = witss.Word("[2^2][1][13^3]")
    alpha = np.pi * np.array([1.5, -.32, .6])

    actual = np.zeros((x.shape[0], ))
    for t in range(x.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_1, 1] ** 2
                        * x[t_2, 0]
                        * x[t_3, 0] * x[t_3, 2] ** 3
                        * np.cos(alpha[2] * (t - t_3) / 50) ** 2
                        * np.cos(alpha[1] * (t_3 - t_2) / 50) ** 2
                        * np.cos(alpha[0] * (t_2 - t_1) / 50) ** 2
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        witss.iss(
            x, word,
            partial=False,
            weighting=witss.weighting.Cosine(alpha, exponent=2, outer=True),
            normalize=True,
        ),
        rtol=1e-4,
    )
