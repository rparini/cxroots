# have a common seed for each testing process
from datetime import date
from math import pi

import numpy as np
import pytest
from numpy.random import uniform

from cxroots.paths import ComplexArc, ComplexLine

today = date.today()
np.random.seed(today.year * today.month * today.day)


@pytest.mark.parametrize("a", uniform(-10, 10, size=3) + 1j * uniform(-10, 10, size=3))
@pytest.mark.parametrize("b", uniform(-10, 10, size=3) + 1j * uniform(-10, 10, size=3))
@pytest.mark.parametrize("p", uniform(-10, 10, size=3) + 1j * uniform(-10, 10, size=3))
def test_distance_line(a, b, p):
    t = np.linspace(0, 1, 100001)
    line = ComplexLine(a, b)
    assert line.distance(p) == pytest.approx(np.min(np.abs(line(t) - p)), abs=1e-6)


@pytest.mark.parametrize("z0", uniform(-10, 10, size=3) + 1j * uniform(-10, 10, size=3))
@pytest.mark.parametrize("r", uniform(0, 10, size=3))
@pytest.mark.parametrize("t0", uniform(0, 2 * pi, size=3))
@pytest.mark.parametrize("dt", uniform(-2 * pi, 2 * pi, size=3))
@pytest.mark.parametrize("p", uniform(-10, 10, size=3) + 1j * uniform(-10, 10, size=3))
def test_distance_arc(z0, r, t0, dt, p):
    z0 = 0
    t = np.linspace(0, 1, 100001)
    arc = ComplexArc(z0, r, t0, dt)
    assert arc.distance(p) == pytest.approx(np.min(np.abs(arc(t) - p)), abs=1e-6)
