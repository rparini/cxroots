import pytest
import numpy as np
from numpy import cos, sin

from cxroots import Rectangle

@pytest.mark.parametrize('useDerivative', [True, False])
def test_count_roots(useDerivative):
	"""
	Example from "Locating all the Zeros of an Analytic Function in one Complex Variable"
	M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2

	There should be 424 roots inside this contour
	"""
	C  = Rectangle([-20.3,20.7], [-20.3,20.7])
	f  = lambda z: z**50 + z**12 - 5*sin(20*z)*cos(12*z) - 1
	df = lambda z: 50*z**49 + 12*z**11 + 60*sin(12*z)*sin(20*z) - 100*cos(12*z)*cos(20*z)

	if not useDerivative: df = None

	assert C.count_roots(f, df, verbose=True) == 424


@pytest.mark.xfail(reason='Possibly lack of high precision arithmetic')
def test_RingOscillator():
	# problem in section 4.3 of [DSZ]
	from cxroots import Rectangle, findRoots

	def A(z):
		t = 2.
		return np.array([[-0.0166689-2.12e-14*z, 1/60. + 6e-16*exp(-t*z)*z],
		 		  		 [0.0166659+exp(-t*z)*(-0.000037485+6e-16*z), -0.0166667-6e-16*z]])

	def dA(z):
		t = 2.
		return np.array([[-2.12e-14*np.ones_like(z), 6e-16*exp(-t*z) - t*6e-16*exp(-t*z)*z],
				   		 [-t*exp(-t*z)*(-0.000037485+6e-16*z)+exp(-t*z)*6e-16, -6e-16*np.ones_like(z)]])
	
	def f(z):
		AVal = np.rollaxis(A(z),-1,0)
		return np.linalg.det(AVal)

	def df(z):
		AVal  = A(z)
		dAVal = dA(z)
		return dAVal[0,0]*AVal[1,1] + AVal[0,0]*dAVal[1,1] - dAVal[0,1]*AVal[1,0] - AVal[0,1]*dAVal[1,0]


	box = Rectangle([-12,0], [-40,40])
	# roots_fdf = findRoots(box, f, df)
	# roots_f = findRoots(box, f)

	# XXX: No roots are recorded within the initial contour.
	# 	Perhaps because the coefficents of z are very small?
	# 	Perhaps need higher precision?
	assert box.count_enclosed_roots(f, df) != 0
	assert box.count_enclosed_roots(f) != 0

	# compare with fig 4 of [DSZ]
	# roots_fdf.show()
