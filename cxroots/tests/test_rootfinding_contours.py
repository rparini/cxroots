import unittest
import numpy as np
from numpy import pi

from cxroots import Circle, Rectangle, AnnulusSector, Annulus
from cxroots.tests.ApproxEqual import roots_approx_equal

class TestRootfindingContours(unittest.TestCase):
	def setUp(self):
		self.roots = roots = [0, -1.234, 1+1j, 1-1j, 2.345]
		self.multiplicities = [1,1,1,1,1]
		self.f  = lambda z: (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4])
		self.df = lambda z: (z-roots[1])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[2])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[3])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[4]) + (z-roots[0])*(z-roots[1])*(z-roots[2])*(z-roots[3])
	
		self.Circle = Circle(0,3)
		self.Rectangle = Rectangle([-2,2],[-2,2])
		self.halfAnnulus = AnnulusSector(0, [0.5,3], [-pi/2, pi/2])
		self.Annulus = Annulus(0, [1,2])

	def test_rootfinding_circle_fdf(self):
		roots_approx_equal(self.Circle.roots(self.f, self.df, verbose=True), (self.roots, self.multiplicities), decimal=7)

	def test_rootfinding_circle_f(self):
		roots_approx_equal(self.Circle.roots(self.f, self.df, verbose=True), (self.roots, self.multiplicities), decimal=7)

	def test_rootfinding_rectangle_fdf(self):
		roots_approx_equal(self.Rectangle.roots(self.f, self.df, verbose=True), (self.roots[:-1], self.multiplicities[:-1]), decimal=7)

	def test_rootfinding_rectangle_f(self):
		roots_approx_equal(self.Rectangle.roots(self.f, self.df, verbose=True), (self.roots[:-1], self.multiplicities[:-1]), decimal=7)

	def test_rootfinding_halfAnnulus_fdf(self):
		roots_approx_equal(self.halfAnnulus.roots(self.f, self.df, verbose=True), (self.roots[2:], self.multiplicities[2:]), decimal=7)

	def test_rootfinding_halfAnnulus_f(self):
		roots_approx_equal(self.halfAnnulus.roots(self.f, self.df, verbose=True), (self.roots[2:], self.multiplicities[2:]), decimal=7)

	def test_rootfinding_Annulus_fdf(self):
		roots_approx_equal(self.Annulus.roots(self.f, self.df, verbose=True), (self.roots[1:-1], self.multiplicities[1:-1]), decimal=7)

	def test_rootfinding_Annulus_f(self):
		roots_approx_equal(self.Annulus.roots(self.f, self.df, verbose=True), (self.roots[1:-1], self.multiplicities[1:-1]), decimal=7)

if __name__ == '__main__':
	unittest.main(verbosity=3)
