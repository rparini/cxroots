import pytest

from cxroots.iterative_methods import secant
from numpy import pi, cos, sin


def test_secant():
    """
    Example from Table 2.5 of "Numerical Analysis" by Richard L. Burden,
    J. Douglas Faires
    """

    def f(x):
        return cos(x) - x

    def df(x):
        return -sin(x) - 1

    iterations = []

    def callback(x, dx, y, iteration):
        return iterations.append(x)

    x, _ = secant(0.5, pi / 4, f, callback=callback)
    iterations.append(x)

    correct_iterations = [
        0.7363841388,
        0.7390581392,
        0.7390851493,
        0.7390851332,
        0.7390851332,
    ]

    assert iterations == pytest.approx(correct_iterations)
