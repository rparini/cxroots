from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate

from cxroots.Contours import Circle, Rectangle

class Derivative(object):
	"""
	Approximation to the derivaive of an analytic function using a Taylor expansion
	"""
	def __init__(self, f, C):
		self.f = f
		self.C = C
		self.z0 = self.C.centerPoint
		self.a_cache = {}

	def a(self, s):
		if s in self.a_cache.keys():
			return self.a_cache[s]
		else:
			integrand = lambda z: self.f(z)/(z-self.z0)**(s+1)
			a = self.C.integrate(integrand, rombergDivMax=int(1e6))/(2j*pi)
			self.a_cache[s] = a
			return a

	def __call__(self, z, errTol=1e-6):
		# err = inf
		# fPrime = 0
		# j = 1
		# while err > errTol:
		# 	nextTerm = j*self.a(j)*z**(j-1)
		# 	err = abs(nextTerm - fPrime)
		# 	fPrime += nextTerm

		# 	print(j, fPrime)

		# 	j += 1

		# return fPrime

		return sum([j*self.a(j)*(z-self.z0)**(j-1) for j in range(20)])

if __name__ == '__main__':
	from numpy import sin, cos
	f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
	df = lambda z: 10*(z**9 - z**4) + cos(z)*cos(z/2) - 0.5*sin(z)*sin(z/2)

	# circle = Circle(0, 2)
	rect = Rectangle([-1.5,1.5],[-2,2])

	fPrime = Derivative(f, rect)

	def fPrimeEuler(z):
		h = 1e-8
		return (f(z+h) - f(z))/h

	import matplotlib.pyplot as plt
	
	# print(df(2))
	# print(fPrime(2))
	# print(fPrimeEuler(2))

	print(fPrime.a(0))
	print(fPrime.a(1))
	# print(fPrime.a(2))
	# print(fPrime.a(3))

