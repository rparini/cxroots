"""
A collection of tests

References
----------
[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable"
	M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
[DL] "A Numerical Method for Locating the Zeros of an Analytic function", 
	L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import sin, cos, exp, pi, cosh, sinh, sqrt

np.set_printoptions(linewidth=1e20)

def rootfinding_RingOscillator():
	# XXX: Not working, function values too close to zero everywhere?
	# problem in section 4.3 of [DSZ]
	from cxroots import Rectangle, findRoots

	def A(z):
		t = 2.
		return np.array([[-0.0166689-2.12e-14*z, 1/60. + 6e-16*exp(-t*z)*z],
		 		  		 [0.0166659+exp(-t*z)*(-0.000037485+6e-16*z), -0.0166667-6e-16*z]])

	def dA(z):
		t = 2.
		return np.array([[-2.12e-14*z/z, 6e-16*exp(-t*z) - t*6e-16*exp(-t*z)*z],
				   		 [-t*exp(-t*z)*(-0.000037485+6e-16*z)+exp(-t*z)*6e-16, -6e-16*z/z]])
	
	def f(z):
		AVal = np.rollaxis(A(z),-1,0)
		return np.linalg.det(AVal)

	def df(z):
		dAVal = np.rollaxis(dA(z),-1,0)
		return np.linalg.det(dAVal)


	box = Rectangle([-12,0], [-40,40])
	roots_fdf = findRoots(box, f, df, integrandUpperBound=1e6)
	# roots_f = findRoots(box, f)

	print(roots_fdf)

	# # XXX: There don't seem to be any roots within the initial contour?
	# # 	Perhaps there is an issue with the coefficents of z being very small?
	# print(box.count_enclosed_roots(f, df, integerTol=1e-2))
	# print(box.count_enclosed_roots(f, reqEqualZeros=10))

	# compare with fig 4 of [DSZ]
	# import matplotlib.pyplot as plt
	# plt.scatter(np.real(roots_fdf), np.imag(roots_fdf), marker='+')
	# plt.scatter(np.real(roots_f), np.imag(roots_f), marker='x')
	# plt.show()

if __name__ == '__main__':
	pass
