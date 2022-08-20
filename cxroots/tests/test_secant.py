import pytest
from numpy import cos, pi

from cxroots.iterative_methods import secant


def test_secant():
    """
    Example from Table 2.5 of "Numerical Analysis" by Richard L. Burden,
    J. Douglas Faires
    """

    def f(x):
        return cos(x) - x

    iterations = []

    def callback(x, dx, y, iteration):
        return iterations.append(x)

    secant(0.5, pi / 4, f, callback=callback)

    correct_iterations = [
        0.7363841388,
        0.7390581392,
        0.7390851493,
        0.7390851332,
        0.7390851332,
    ]

    assert iterations == pytest.approx(correct_iterations)
