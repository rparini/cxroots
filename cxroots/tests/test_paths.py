import pytest

import numpy as np
from numpy import pi
from numpy.random import uniform

from cxroots.Paths import ComplexArc, ComplexLine

# have a common seed for each testing process
from datetime import date
today = date.today()
np.random.seed(today.year*today.month*today.day)

@pytest.mark.parametrize('a', uniform(-10, 10, size=3) + 1j*uniform(-10, 10, size=3))
@pytest.mark.parametrize('b', uniform(-10, 10, size=3) + 1j*uniform(-10, 10, size=3))
@pytest.mark.parametrize('P', uniform(-10, 10, size=3) + 1j*uniform(-10, 10, size=3))
def test_distance_line(a, b, P):
	t = np.linspace(0,1,100001)
	C = ComplexLine(a, b)
	assert C.distance(P) == pytest.approx(np.min(np.abs(C(t) - P)), abs=1e-6)


@pytest.mark.parametrize('z0', uniform(-10, 10, size=3) + 1j*uniform(-10, 10, size=3))
@pytest.mark.parametrize('R', uniform(0, 10, size=3))
@pytest.mark.parametrize('t0', uniform(0, 2*pi, size=3))
@pytest.mark.parametrize('dt', uniform(-2*pi, 2*pi, size=3))
@pytest.mark.parametrize('P', uniform(-10, 10, size=3) + 1j*uniform(-10, 10, size=3))
def test_distance_arc(z0, R, t0, dt, P):
	z0 = 0
	t = np.linspace(0,1,100001)
	C = ComplexArc(z0, R, t0, dt)
	assert C.distance(P) == pytest.approx(np.min(np.abs(C(t) - P)), abs=1e-6)
