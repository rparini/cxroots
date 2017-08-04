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

def numberOfRoots_DellnitzSchutzeZheng_fdf():
	from cxroots import Rectangle

	rectangle = Rectangle([-20.3,20.7], [-20.3,20.7])

	f = lambda z: z**50 + z**12 - 5*sin(20*z)*cos(12*z) - 1
	df = lambda z: 50*z**49 + 12*z**11 + 60*sin(12*z)*sin(20*z) - 100*cos(12*z)*cos(20*z)
	
	enclosedZeros = rectangle.count_roots(f, df)

	# should be 424 roots according to [DSZ]
	print('enclosed zeros with f and df: %i'%enclosedZeros, 'should be 424')
	
def numberOfRoots_DellnitzSchutzeZheng_f():
	from cxroots import Rectangle

	rectangle = Rectangle([-20.3,20.7], [-20.3,20.7])
	f = lambda z: z**50 + z**12 - 5*sin(20*z)*cos(12*z) - 1

	# require a higher number of equal consecutive evaluations of the number of roots than by default
	enclosedZeros = rectangle.count_roots(f)

	# should be 424 roots according to [DSZ]
	print('enclosed zeros with just f: %i'%enclosedZeros, 'should be 424')

def rootfinding_AnnularCombustionChamber():
	# problem in section 4.2 of [DSZ]
	from cxroots import Rectangle, findRoots

	A = -0.19435
	B = 1000.41
	C = 522463
	T = 0.005

	f = lambda z: z**2 + A*z + B*exp(-T*z) + C
	df = lambda z: 2*z + A - B*T*exp(-T*z)

	rectangle = Rectangle([-15000,5000], [-15000,15000])
	roots_fdf = findRoots(rectangle, f, df)
	roots_f = findRoots(rectangle, f)

	# compare with fig 3 of [DSZ]
	import matplotlib.pyplot as plt
	plt.scatter(np.real(roots_fdf), np.imag(roots_fdf), marker='+')
	plt.scatter(np.real(roots_f), np.imag(roots_f), marker='x')
	plt.show()

def rootfinding_RingOscillator():
	# XXX: Not working
	# problem in section 4.3 of [DSZ]
	from cxroots import Rectangle, findRoots

	t = 2.
	A = lambda z: np.array([[-0.0166689-2.12e-14*z, 1/60. + 6e-16*exp(-t*z)*z],
		 		  			[0.0166659+exp(-t*z)*(-0.000037485+6e-16*z), -0.0166667-6e-16*z]])
	dA = lambda z: np.array([[-2.12e-14, 6e-16*exp(-t*z) - t*6e-16*exp(-t*z)*z],
				   			 [-t*exp(-t*z)*(-0.000037485+6e-16*z)+exp(-t*z)*6e-16, -6e-16]])
	
	def f(z):
		AVal = np.rollaxis(A(z),-1,0)
		return np.linalg.det(AVal)

	def df(z):
		dAVal = np.rollaxis(dA(z),-1,0)
		return np.linalg.det(dAVal)

	box = Rectangle([-12,0], [-40,40])
	roots_fdf = findRoots(box, f, df, integerTol=1e-2)
	roots_f = findRoots(box, f, reqEqualZeros=10)

	# XXX: There don't seem to be any roots within the initial contour?
	# 	Perhaps there is an issue with the coefficents of z being very small?
	print(box.count_enclosed_roots(f, df, integerTol=1e-2))
	print(box.count_enclosed_roots(f, reqEqualZeros=10))

	# compare with fig 4 of [DSZ]
	# import matplotlib.pyplot as plt
	# plt.scatter(np.real(roots_fdf), np.imag(roots_fdf), marker='+')
	# plt.scatter(np.real(roots_f), np.imag(roots_f), marker='x')
	# plt.show()

def rootfinding_polynomial():
	from cxroots import Circle, Rectangle, PolarRect, findRoots

	roots = [0, 1-1j, 1+1j, -1.234, 2.345]

	f = lambda z: (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4])
	df = lambda z: (z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])

	# try different contours to make sure we are getting the right roots
	# unlike the proposal in [DL] there is no 'buffer zone'.  Only the user specified
	# region is searched

	circle = Circle(0,3)
	print(findRoots(circle, f))
	print(findRoots(circle, f, df))

	rectangle = Rectangle([-1,3],[-2,2])
	print(findRoots(rectangle, f))
	print(findRoots(rectangle, f, df))

	halfAnnulus = PolarRect(0, [0.5,3], [-pi/2, pi/2])
	print(findRoots(halfAnnulus, f))
	print(findRoots(halfAnnulus, f, df))

def rootfinding_realCoeffPoly():
	from cxroots import Circle, Rectangle, PolarRect, findRoots, showRoots

	circle = Circle(0, 1.5)

	# time it
	f = lambda z: z**27-2*z**11+0.5*z**6-1
	df = lambda z: 27*z**26-22*z**10+3*z**5

	# findRoots(circle, f, df)

	conjugateSymmetry = lambda z: [z.conjugate()]
	showRoots(circle, f, df, guessRootSymmetry = conjugateSymmetry)

def test_newton():
	from cxroots.IterativeMethods import newton

	# example from Table 2.4 of "Numerical Analysis" by Richard L. Burden, J. Douglas Faires
	answer = [0.7395361337,
			  0.7390851781,
			  0.7390851332,
			  0.7390851332]

	f  = lambda x: cos(x)-x
	df = lambda x: -sin(x)-1
	callback = lambda x, dx, y, iteration: print(iteration, x, dx, x-answer[iteration-1])
	
	print(newton(pi/4, f, df, callback=callback))

def test_secant():
	from cxroots.IterativeMethods import secant

	# example from Table 2.5 of "Numerical Analysis" by Richard L. Burden, J. Douglas Faires
	answer = [0.7363841388,
			  0.7390581392,
			  0.7390851493,
			  0.7390851332,
			  0.7390851332]

	f  = lambda x: cos(x)-x
	df = lambda x: -sin(x)-1
	callback = lambda x, dx, y, iteration: print(iteration, x, dx, x-answer[iteration-1])
	
	print(secant(0.5, pi/4, f, callback=callback))

def simple_test():
	from cxroots import Rectangle, showRoots, demo_findRoots, findRoots
	from numpy import pi, sin, cos
	import numpy as np

	rect = Rectangle([-2,2],[-2,2])
	# f  = lambda z: z*(z**10 - 2*z**5 + sin(z)*cos(z/2))8
	# df = lambda z: z*(10*z**9 - 10*z**4 + cos(z)*cos(z/2) - 0.5*sin(z)*sin(z/2)) + z**10 - 2*z**5 + sin(z)*cos(z/2)

	f = lambda z: z**3 * (z-1.2)**2
	df = lambda z: 3*(z)**2 * (z-1.2)**2 + 2*z**3 * (z-1.2)

	roots, multiplicities = findRoots(rect, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)
	# demo_findRoots(rect, f, df, absTol=1e-8, relTol=1e-8)
	# showRoots(rect, f, df)

def ex1():
	# Ex 1.4.1 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Circle, findRoots, demo_findRoots
	C = Circle(0,3)

	e = 1e-2
	f  = lambda z: (z-e)*(1+(z-sqrt(3))**2)
	df = lambda z: (1+(z-sqrt(3))**2) + (z-e)*2*(z-sqrt(3))

	roots, multiplicities = findRoots(C, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex2():
	# Ex 1.4.2 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Circle, findRoots, demo_findRoots
	C = Circle(0,2)

	f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
	df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

	roots, multiplicities = findRoots(C, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex2b():
	# Ex 1.4.2 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-2,2],[-2,2])

	f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
	df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

	roots, multiplicities = findRoots(C, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex3():
	# Ex 1.4.3 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Circle, findRoots, demo_findRoots
	C = Circle(0,5)

	f  = lambda z: z**2*(z-1)*(z-2)*(z-3)*(z-4)+z*sin(z)
	df = lambda z: 2*z*(3*z**4-25*z**3+70*z**2-75*z+24)+sin(z)+z*cos(z)

	roots, multiplicities = findRoots(C, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex4():
	# Ex 1.4.4 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Circle, findRoots, demo_findRoots
	C = Circle(0,3)

	f  = lambda z: (z*(z-2))**2*(exp(2*z)*cos(z)+z**3-1-sin(z))
	df = lambda z: 2*z*(z-2)**2*(exp(2*z)*cos(z)+z**3-1-sin(z))+2*(z-2)*z**2*(exp(2*z)*cos(z)+z**3-1-sin(z))+(z*(z-2))**2*(2*exp(2*z)*cos(z)-exp(2*z)*sin(z)+3*z**2-cos(z))

	roots, multiplicities = findRoots(C, f, df)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex5():
	# Ex 1.4.5 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Circle, findRoots, demo_findRoots
	C = Circle(0,11)

	f  = lambda z: np.prod([z-k for k in range(1,11)], axis=0)
	df = lambda z: np.sum([np.prod([z-k for k in range(1,11) if k!=m], axis=0) for m in range(1,11)], axis=0)

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex5b():
	# Ex 1.4.5 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-1,11],[-1,1])

	f  = lambda z: np.prod([z-k for k in range(1,11)], axis=0)
	df = lambda z: np.sum([np.prod([z-k for k in range(1,11) if k!=m], axis=0) for m in range(1,11)], axis=0)

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex_ZEAL1a():
	# Ex 1.5.1 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-2,2], [-2,3])

	f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
	df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12, M=5)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex_ZEAL1b():
	# Ex 1.5.1 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	# This time with M=2
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-2,2], [-2,3])

	f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
	df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12, M=2)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex_ZEAL2():
	# Ex 1.5.2 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-0.5,5.5], [-0.5,1.5])

	f  = lambda z: z**2*(z-1)*(z-2)*(z-3)*(z-4)+z*sin(z)
	df = lambda z: 2*z*(3*z**4-25*z**3+70*z**2-75*z+24)+sin(z)+z*cos(z)

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12, M=5)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

def ex_ZEAL3():
	# Ex 1.5.3 from "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	from cxroots import Rectangle, findRoots, demo_findRoots
	C = Rectangle([-1,3], [-1,1])

	f  = lambda z: (z*(z-2))**2 * (exp(2*z)*cos(z)+z**3-1-sin(z))
	df = lambda z: 2*z*(z-2)**2 * (exp(2*z)*cos(z)+z**3-1-sin(z)) + 2*z**2*(z-2) * (exp(2*z)*cos(z)+z**3-1-sin(z)) + (z*(z-2))**2 * (2*exp(2*z)*cos(z)-exp(2*z)*sin(z)+3*z**2-cos(z))

	roots, multiplicities = findRoots(C, f, df, absTol=1e-12, relTol=1e-12, M=5)
	print('----- Roots -----')
	for root, multiplicity in zip(roots, multiplicities):
		print(multiplicity, root)

if __name__ == '__main__':
	#### rootfinding_RingOscillator(), XXX: Not working

	# numberOfRoots_DellnitzSchutzeZheng_fdf()
	# numberOfRoots_DellnitzSchutzeZheng_f()

	# rootfinding_AnnularCombustionChamber()

	# rootfinding_polynomial()
	# rootfinding_realCoeffPoly()

	# print('-- Newton --')
	# test_newton()
	# print('-- Secant --')
	# test_secant()

	# ex2b()

