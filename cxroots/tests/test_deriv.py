import numpy as np
import pytest
from numpy import cos, sin

from cxroots import Circle, Rectangle, cx_derivative


@pytest.mark.parametrize(
    "contour",
    [
        pytest.param(Circle(0, 2), id="circle"),
        pytest.param(Rectangle([-1.5, 1.5], [-2, 2]), id="rect"),
        pytest.param(None, id="default"),
    ],
)
def test_cx_derivative(contour):
    def f(z):
        return z**10 - 2 * z**5 + sin(z) * cos(z / 2)

    def df(z):
        return 10 * (z**9 - z**4) + cos(z) * cos(z / 2) - 0.5 * sin(z) * sin(z / 2)

    z = np.array([-1.234, 0.3 + 1j, 0.1j, -0.9 - 0.5j])

    assert cx_derivative(f, z, n=1, contour=contour) == pytest.approx(df(z))
