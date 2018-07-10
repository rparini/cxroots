import numpy as np
from numpy.random import uniform
import pytest

from cxroots.Paths import ComplexArc, ComplexLine

@pytest.mark.parametrize('a', uniform(-10, 10, size=5) + 1j*uniform(-10, 10, size=5))
@pytest.mark.parametrize('b', uniform(-10, 10, size=5) + 1j*uniform(-10, 10, size=5))
@pytest.mark.parametrize('P', uniform(-10, 10, size=5) + 1j*uniform(-10, 10, size=5))
def test_distance_line(a, b, P):
	t = np.linspace(0,1,100000)
	line = ComplexLine(a, b)
	assert line.distance(P) == pytest.approx(np.min(np.abs(line(t) - P)), 1e-6)