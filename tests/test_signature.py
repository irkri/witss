import numpy as np

import iss


def test_iss() -> None:
    x = np.random.normal(size=(100, 3))
    word = iss.Word("[12^3][32][11]")

    actual = np.zeros((100, ))

    for t in range(100):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    actual[t] += (
                          x[t_3, 0] ** 2
                        * x[t_2, 1] * x[t_2, 2]
                        * x[t_1, 0] * x[t_1, 1] ** 3
                    )

    np.testing.assert_allclose(actual, iss.iss(x, word))
