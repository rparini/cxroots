import unittest
import numpy as np
from scipy import pi

from cxroots import Circle, Rectangle, PolarRect
from cxroots.tests.SetsApproxEqual import sets_approx_equal

class TestSimpleRootfindingPolynomial(unittest.TestCase):
	"""
	Rootfinding but only simple roots
	"""

	def setUp(self):
		self.roots = roots = [-1.234, 0,  1+1j, 1-1j, 2.345]
		self.f  = lambda z: (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4])
		self.df = lambda z: (z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])
	
		self.Circle = Circle(0,3)
		self.Rectangle = Rectangle([-2,2],[-2,2])
		self.halfAnnulus = PolarRect(0, [0.5,3], [-pi/2, pi/2])

	def test_rootfindingPolynomial_circle_fdf(self):
		approxRoots, multiplicities = self.Circle.roots(self.f, self.df)
		sets_approx_equal(approxRoots, self.roots, decimal=7)

	def test_rootfindingPolynomial_circle_f(self):
		approxRoots, multiplicities = self.Circle.roots(self.f)
		sets_approx_equal(approxRoots, self.roots, decimal=7)

	def test_rootfindingPolynomial_rectangle_fdf(self):
		approxRoots, multiplicities = self.Rectangle.roots(self.f, self.df)
		sets_approx_equal(approxRoots, self.roots[:-1], decimal=7)

	def test_rootfindingPolynomial_rectangle_f(self):
		approxRoots, multiplicities = self.Rectangle.roots(self.f)
		sets_approx_equal(approxRoots, self.roots[:-1], decimal=7)

	def test_rootfindingPolynomial_halfAnnulus_fdf(self):
		approxRoots, multiplicities = self.halfAnnulus.roots(self.f, self.df)
		sets_approx_equal(approxRoots, self.roots[2:], decimal=7)

	def test_rootfindingPolynomial_halfAnnulus_f(self):
		approxRoots, multiplicities = self.halfAnnulus.roots(self.f)
		sets_approx_equal(approxRoots, self.roots[2:], decimal=7)

if __name__ == '__main__':
	unittest.main(verbosity=3)
