import numpy as np
import pytest
from numpy import cos, exp, sin

from cxroots import Rectangle


@pytest.mark.parametrize("use_df", [True, False])
def test_count_roots(use_df):
    """
    From "Locating all the Zeros of an Analytic Function in one Complex Variable"
    M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2

    There should be 424 roots inside this contour
    """
    contour = Rectangle([-20.3, 20.7], [-20.3, 20.7])

    def f(z):
        return z**50 + z**12 - 5 * sin(20 * z) * cos(12 * z) - 1

    if use_df:

        def df(z):
            return (
                50 * z**49
                + 12 * z**11
                + 60 * sin(12 * z) * sin(20 * z)
                - 100 * cos(12 * z) * cos(20 * z)
            )

    else:
        df = None

    assert contour.count_roots(f, df) == 424


@pytest.mark.xfail(reason="Possibly lack of high precision arithmetic")
def test_ring_oscillator():
    """Problem in section 4.3 of [DSZ]"""

    def A(z):  # noqa: N802
        t = 2.0
        return np.array(
            [
                [-0.0166689 - 2.12e-14 * z, 1 / 60.0 + 6e-16 * exp(-t * z) * z],
                [
                    0.0166659 + exp(-t * z) * (-0.000037485 + 6e-16 * z),
                    -0.0166667 - 6e-16 * z,
                ],
            ]
        )

    def dA(z):  # noqa: N802
        t = 2.0
        return np.array(
            [
                [
                    -2.12e-14 * np.ones_like(z),
                    6e-16 * exp(-t * z) - t * 6e-16 * exp(-t * z) * z,
                ],
                [
                    -t * exp(-t * z) * (-0.000037485 + 6e-16 * z) + exp(-t * z) * 6e-16,
                    -6e-16 * np.ones_like(z),
                ],
            ]
        )

    def f(z):
        A_val = np.rollaxis(A(z), -1, 0)  # noqa: N806
        return np.linalg.det(A_val)

    def df(z):
        A_val = A(z)  # noqa: N806
        dA_val = dA(z)  # noqa: N806
        return (
            dA_val[0, 0] * A_val[1, 1]
            + A_val[0, 0] * dA_val[1, 1]
            - dA_val[0, 1] * A_val[1, 0]
            - A_val[0, 1] * dA_val[1, 0]
        )

    box = Rectangle([-12, 0], [-40, 40])

    # XXX: No roots are recorded within the initial contour.
    #   Perhaps because the coefficents of z are very small?
    #   Perhaps need higher precision?
    assert box.count_roots(f, df) != 0
    assert box.count_roots(f) != 0

    # roots_fdf = findRoots(box, f, df)
    # roots_f = findRoots(box, f)

    # compare with fig 4 of [DSZ]
    # roots_fdf.show()
