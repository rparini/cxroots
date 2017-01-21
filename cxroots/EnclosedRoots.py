from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate

from cxroots.Contours import Circle, Rectangle

def enclosed_roots(C, f, df=None, integerTol=0.1, nofPrimeMethod='taylor'):
	N = 1
	fVal  = [None]*len(C.segments)
	dfVal = [None]*len(C.segments)
	I = integerTol*1e6j # just to make sure the while loop is run at least once

	nofPrime = False
	if df is None:
		nofPrime = True

	oldI = 0
	while abs(oldI - I) > integerTol/2.:
		oldI = I
		N = 2*N

		t = np.linspace(0,1,N+1)
		dt = t[1]-t[0]

		# store function evaluations
		for i, segment in enumerate(C.segments):
			z = segment(t)
			if fVal[i] is None:
				fVal[i] = f(z)
			else:
				newfVal = np.zeros_like(t, dtype=np.complex128)
				newfVal[::2] = fVal[i]
				newfVal[1::2] = f(z[1::2])
				fVal[i] = newfVal

		if nofPrime:
			if nofPrimeMethod == 'taylor':
				# use available function evaluations to approximate df
				z0 = C.centerPoint

				M = 20 # number of terms in the Taylor expansion for df

				a = []
				for s in range(M):
					a_s = 0
					for i, segment in enumerate(C.segments):
						integrand = fVal[i]/(segment(t)-z0)**(s+1)*segment.dzdt(t)
						
						# romberg integration on a set of sample points
						a_s += scipy.integrate.romb(integrand, dx=dt)/(2j*pi)
					a.append(a_s)

				df = lambda z: sum(j*a[j]*(z-z0)**(j-1) for j in range(M))

		# store derivative evaluations.
		for i, segment in enumerate(C.segments):
			z = segment(t)
			if dfVal[i] is None:
				dfVal[i] = df(z)
			else:
				newdfVal = np.zeros_like(t, dtype=np.complex128)
				newdfVal[::2] = dfVal[i]
				newdfVal[1::2] = df(z[1::2])
				dfVal[i] = newdfVal

		I = sum(scipy.integrate.romb(dfVal[i]/fVal[i]*segment.dzdt(t), dx=dt) for i, segment in enumerate(C.segments))/(2j*pi)

	numberOfZeros = int(round(I.real))
	if numberOfZeros < 0 or abs(I.real - numberOfZeros) > integerTol or abs(I.imag) > integerTol:
		raise RuntimeError('The integral %s is not sufficiently close to a positive integer'%integral)

	return numberOfZeros


if __name__ == '__main__':
	from numpy import sin, cos
	f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
	df = lambda z: 10*(z**9 - z**4) + cos(z)*cos(z/2) - 0.5*sin(z)*sin(z/2)

	rect = Rectangle([-1.5,1.5],[-2,2])
	circle = Circle(0,2)

	# print(rect.enclosed_zeros(f, df))
	# print(rect.enclosed_zeros(f))
	# print(circle.enclosed_zeros(f))

	# print(enclosed_roots(rect, f, df))
	print(enclosed_roots(rect, f))