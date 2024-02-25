import numpy as np
import pytest
from numpy import cos, sin

from cxroots.derivative import central_diff


def test_central_diff():
    def f(z):
        return z**10 - 2 * z**5 + sin(z) * cos(z / 2)

    def df(z):
        return 10 * (z**9 - z**4) + cos(z) * cos(z / 2) - 0.5 * sin(z) * sin(z / 2)

    z = np.array([-1.234, 0.3 + 1j, 0.1j, -0.9 - 0.5j])

    approx_df = central_diff(f)

    assert approx_df(z) == pytest.approx(df(z), abs=1e-8)
