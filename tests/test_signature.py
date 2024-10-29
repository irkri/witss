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

    np.testing.assert_allclose(actual, iss.iss(x, word, strict=False))

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

    np.testing.assert_allclose(actual, iss.iss(x, word, normalize=True))

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
        iss.iss(x, word, normalize=True, strict=False),
    )


def test_partial() -> None:
    x = np.random.normal(size=(100, 3))
    word = iss.Word("[12^3][32][12]")

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

    np.testing.assert_allclose(actual, iss.iss(x, word, partial=True).numpy())

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
        iss.iss(x, word, partial=True, normalize=True).numpy(),
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
        iss.iss(x, word, partial=True, strict=False).numpy(),
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
        iss.iss(x, word, partial=True, strict=False, normalize=True).numpy(),
    )


def test_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = iss.Word("[13][2^2][1]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Exponential(alpha, outer=False),
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Exponential(alpha, outer=False),
            normalize=True,
        ),
    )



def test_outer_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = iss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((50, ))

    for t in range(50):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 50)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Exponential(alpha, outer=True),
        ),
    )

    actual = np.zeros((50, ))

    for t in range(50):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 50)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Exponential(alpha, outer=True),
            normalize=True,
        ),
    )


def test_partial_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = iss.Word("[12^3][32][12]")
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
        iss.iss(
            x, word,
            partial=True,
            weighting=iss.weighting.Exponential(alpha, outer=False),
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
        iss.iss(
            x, word,
            partial=True,
            weighting=iss.weighting.Exponential(alpha, outer=False),
            normalize=True,
        ).numpy(),
    )


def test_partial_outer_exp() -> None:
    x = np.random.normal(size=(50, 3))
    word = iss.Word("[12^3][32][12]")
    alpha = np.array([.4, .8, 2])

    actual = np.zeros((3, 50))

    for t in range(50):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                * np.exp(-alpha[0] * (t - t_3) / 50)
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    * np.exp(-alpha[1] * (t - t_3) / 50)
                    * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 50)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=True,
            weighting=iss.weighting.Exponential(alpha),
        ).numpy(),
    )

    actual = np.zeros((3, 50))

    for t in range(50):
        for t_3 in range(t+1):
            actual[0, t] += (
                x[t_3, 0] * x[t_3, 1] ** 3
                * np.exp(-alpha[0] * (t - t_3) / 50)
                / (t+1)
            )
            for t_2 in range(t_3):
                actual[1, t] += (
                    x[t_2, 0] * x[t_2, 1] ** 3
                    * x[t_3, 1] * x[t_3, 2]
                    * np.exp(-alpha[1] * (t - t_3) / 50)
                    * np.exp(-alpha[0] * (t_3 - t_2) / 50)
                    / (t) / (t_3)
                )
                for t_1 in range(t_2):
                    actual[2, t] += (
                          x[t_3, 0] * x[t_3, 1]
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                        * np.exp(-alpha[2] * (t - t_3) / 50)
                        * np.exp(-alpha[1] * (t_3 - t_2) / 50)
                        * np.exp(-alpha[0] * (t_2 - t_1) / 50)
                        / (t-1) / (t_3-1) / (t_2)
                    )

    np.testing.assert_allclose(
        actual,
        iss.iss(
            x, word,
            partial=True,
            weighting=iss.weighting.Exponential(alpha),
            normalize=True,
        ).numpy(),
    )


def test_cos() -> None:
    x = np.random.random((50, 3))
    word = iss.Word("[12][2][33]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=1, outer=False),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = iss.Word("[2^2][1][13^3]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=2, outer=False),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = iss.Word("[12][2][33]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=1, outer=False),
            normalize=True,
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = iss.Word("[2^2][1][13^3]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=2, outer=False),
            normalize=True,
        ),
        rtol=1e-4,
    )

def test_cos_outer() -> None:
    x = np.random.random((50, 3))
    word = iss.Word("[12][2][33]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=1, outer=True),
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = iss.Word("[2^2][1][13^3]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=2, outer=True),
        ),
        rtol=1e-4,
    )


    x = np.random.random((50, 3))
    word = iss.Word("[12][2][33]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=1, outer=True),
            normalize=True,
        ),
        rtol=1e-4,
    )

    x = np.random.random((50, 3))
    word = iss.Word("[2^2][1][13^3]")
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
        iss.iss(
            x, word,
            partial=False,
            weighting=iss.weighting.Cosine(alpha, exponent=2, outer=True),
            normalize=True,
        ),
        rtol=1e-4,
    )


if __name__ == "__main__":
    test_base()
