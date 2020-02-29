import pytest
import numpy as np
from numpy import cos, sin

from cxroots import Circle, Rectangle
from cxroots import CxDerivative

@pytest.mark.parametrize('C', [
	pytest.param(Circle(0, 2), id='circle'),
	pytest.param(Rectangle([-1.5,1.5],[-2,2]), id='rect'),
	pytest.param(None, id='default')
])
def test_CxDerivative(C):
	f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
	df = lambda z: 10*(z**9 - z**4) + cos(z)*cos(z/2) - 0.5*sin(z)*sin(z/2)

	z = np.array([-1.234, 0.3+1j, 0.1j, -0.9-0.5j])

	assert CxDerivative(f, z, n=1, contour=C) == pytest.approx(df(z))

