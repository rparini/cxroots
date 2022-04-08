import pytest

from cxroots.IterativeMethods import secant
from numpy import pi, cos, sin


def test_secant():
    # example from Table 2.5 of "Numerical Analysis" by Richard L. Burden, J. Douglas Faires
    f = lambda x: cos(x) - x
    df = lambda x: -sin(x) - 1

    iterations = []
    callback = lambda x, dx, y, iteration: iterations.append(x)
    x, err = secant(0.5, pi / 4, f, callback=callback)
    iterations.append(x)

    correct_iterations = [
        0.7363841388,
        0.7390581392,
        0.7390851493,
        0.7390851332,
        0.7390851332,
    ]

    assert iterations == pytest.approx(correct_iterations)
