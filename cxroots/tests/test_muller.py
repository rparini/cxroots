import pytest

from cxroots.IterativeMethods import muller
from numpy import pi, cos


def test_muller():
    def f(x):
        return cos(x) - x**2 + 1j * x**3

    x, _ = muller(0.5, pi / 4, 0.6, f)
    root = 0.7296100078977741539157356847 + 0.1570923181734581909733787621j
    assert x == pytest.approx(root)
