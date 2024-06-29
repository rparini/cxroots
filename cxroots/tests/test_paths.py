# have a common seed for each testing process
from math import pi

import numpy as np
import pytest

from cxroots.paths import ComplexArc, ComplexLine

rng = np.random.default_rng(159)


@pytest.mark.parametrize(
    "a", rng.uniform(-10, 10, size=3) + 1j * rng.uniform(-10, 10, size=3)
)
@pytest.mark.parametrize(
    "b", rng.uniform(-10, 10, size=3) + 1j * rng.uniform(-10, 10, size=3)
)
@pytest.mark.parametrize(
    "p", rng.uniform(-10, 10, size=3) + 1j * rng.uniform(-10, 10, size=3)
)
def test_distance_line(a, b, p):
    t = np.linspace(0, 1, 100001)
    line = ComplexLine(a, b)
    assert line.distance(p) == pytest.approx(np.min(np.abs(line(t) - p)), abs=1e-6)


@pytest.mark.parametrize(
    "z0", rng.uniform(-10, 10, size=3) + 1j * rng.uniform(-10, 10, size=3)
)
@pytest.mark.parametrize("r", rng.uniform(0, 10, size=3))
@pytest.mark.parametrize("t0", rng.uniform(0, 2 * pi, size=3))
@pytest.mark.parametrize("dt", rng.uniform(-2 * pi, 2 * pi, size=3))
@pytest.mark.parametrize(
    "p", rng.uniform(-10, 10, size=3) + 1j * rng.uniform(-10, 10, size=3)
)
def test_distance_arc(z0, r, t0, dt, p):
    z0 = 0
    t = np.linspace(0, 1, 100001)
    arc = ComplexArc(z0, r, t0, dt)
    assert arc.distance(p) == pytest.approx(np.min(np.abs(arc(t) - p)), abs=1e-6)
