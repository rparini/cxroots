"""
References
----------
[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable"
	M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
[DL] "A Numerical Method for Locating the Zeros of an Analytic function", 
	L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
[KB] "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
"""

import unittest
import numpy as np
from scipy import pi, sqrt, exp, sin, cos

from cxroots import Circle, Rectangle
from cxroots.tests.ApproxEqual import roots_approx_equal

class TestRootfinding(unittest.TestCase):
	def test_rootfinding_141(self):
		# Ex 1.4.1 from [KB]
		C = Circle(0,3)
		e = 1e-2
		f  = lambda z: (z-e)*(1+(z-sqrt(3))**2)
		df = lambda z: (1+(z-sqrt(3))**2) + (z-e)*2*(z-sqrt(3))

		roots = [e, sqrt(3)+1j, sqrt(3)-1j]
		multiplicities = [1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_142(self):
		# Ex 1.4.2 from [KB]
		C = Circle(0,2)
		f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
		df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

		roots = [0,
				 -1.844233953262213, 
				 0.5308949302929305 + 1.33179187675112098j,
				 0.5308949302929305 - 1.33179187675112098j]
		multiplicities = [1,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_142b(self):
		# Ex 1.4.2 from [KB] with a rectangular initial contour
		C = Rectangle([-2,2],[-2,2])
		f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
		df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

		roots = [0,
				 -1.844233953262213, 
				 0.5308949302929305 + 1.33179187675112098j,
				 0.5308949302929305 - 1.33179187675112098j]
		multiplicities = [1,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_143(self):
		# Ex 1.4.3 from [KB]
		C = Circle(0,5)
		f  = lambda z: z**2*(z-1)*(z-2)*(z-3)*(z-4)+z*sin(z)
		df = lambda z: 2*z*(3*z**4-25*z**3+70*z**2-75*z+24)+sin(z)+z*cos(z)

		roots = [0,
				 1.18906588973011365517521756, 
				 1.72843498616506284043592924,
				 3.01990732809571222812005354,
				 4.03038191606046844562845941]
		multiplicities = [2,1,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_144(self):
		# Ex 1.4.4 from [KB]
		C = Circle(0,3)
		f  = lambda z: (z*(z-2))**2*(exp(2*z)*cos(z)+z**3-1-sin(z))
		df = lambda z: 2*z*(z-2)**2*(exp(2*z)*cos(z)+z**3-1-sin(z))+2*(z-2)*z**2*(exp(2*z)*cos(z)+z**3-1-sin(z))+(z*(z-2))**2*(2*exp(2*z)*cos(z)-exp(2*z)*sin(z)+3*z**2-cos(z))

		roots = [-0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
				 -0.4607141197289707542294459477 + 0.6254277693477682516688207854j,
				 0,
				 2,
				 1.66468286974551654134568653]
		multiplicities = [1,1,3,2,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_145(self):
		# Ex 1.4.5 from [KB]
		C = Circle(0,11)
		f  = lambda z: np.prod([z-k for k in range(1,11)], axis=0)
		df = lambda z: np.sum([np.prod([z-k for k in range(1,11) if k!=m], axis=0) for m in range(1,11)], axis=0)

		roots = [1,2,3,4,5,6,7,8,9,10]
		multiplicities = np.ones(10)

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_145b(self):
		# Ex 1.4.5 from [KB] with a rectangular initial contour
		C = Rectangle([-1,11],[-1,1])
		f  = lambda z: np.prod([z-k for k in range(1,11)], axis=0)
		df = lambda z: np.sum([np.prod([z-k for k in range(1,11) if k!=m], axis=0) for m in range(1,11)], axis=0)

		roots = [1,2,3,4,5,6,7,8,9,10]
		multiplicities = np.ones(10)

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_151(self):
		# Ex 1.5.1 from [KB]
		C = Rectangle([-2,2], [-2,3])
		f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
		df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

		roots = [-1.84423395326221337491592440,
				 0,
				 0.5308949302929305274642203840 - 1.331791876751120981651544228j,
				 0.5308949302929305274642203840 + 1.331791876751120981651544228j]
		multiplicities = [1,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_151b(self):
		# Ex 1.5.1 from [KB] with M=2
		from cxroots import Rectangle, findRoots, demo_findRoots
		C = Rectangle([-2,2], [-2,3])
		f  = lambda z: exp(3*z) + 2*z*cos(z) - 1
		df = lambda z: 3*exp(3*z) + 2*cos(z) - 2*z*sin(z)

		roots = [-1.84423395326221337491592440,
				 0,
				 0.5308949302929305274642203840 - 1.331791876751120981651544228j,
				 0.5308949302929305274642203840 + 1.331791876751120981651544228j]
		multiplicities = [1,1,1,1]

		roots_approx_equal(C.roots(f, df, M=2), (roots, multiplicities), decimal=12)

	def test_rootfinding_152(self):
		# Ex 1.5.2 from [KB]
		C = Rectangle([-0.5,5.5], [-0.5,1.5])
		f  = lambda z: z**2*(z-1)*(z-2)*(z-3)*(z-4)+z*sin(z)
		df = lambda z: 2*z*(3*z**4-25*z**3+70*z**2-75*z+24)+sin(z)+z*cos(z)

		roots = [0,
				 1.18906588973011365517521756,
				 1.72843498616506284043592924,
				 3.01990732809571222812005354,
				 4.03038191606046844562845941]
		multiplicities = [2,1,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)

	def test_rootfinding_153(self):
		# Ex 1.5.3 from [KB]
		C = Rectangle([-1,3], [-1,1])
		f  = lambda z: (z*(z-2))**2 * (exp(2*z)*cos(z)+z**3-1-sin(z))
		df = lambda z: 2*z*(z-2)**2 * (exp(2*z)*cos(z)+z**3-1-sin(z)) + 2*z**2*(z-2) * (exp(2*z)*cos(z)+z**3-1-sin(z)) + (z*(z-2))**2 * (2*exp(2*z)*cos(z)-exp(2*z)*sin(z)+3*z**2-cos(z))

		roots = [0, 
				 2,
				 1.66468286974551654134568653,
				 -0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
				 -0.4607141197289707542294459477 + 0.6254277693477682516688207854j]
		multiplicities = [3,2,1,1,1]

		roots_approx_equal(C.roots(f, df), (roots, multiplicities), decimal=12)



if __name__ == '__main__':
	unittest.main(verbosity=3)
