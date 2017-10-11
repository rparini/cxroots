import unittest
import numpy as np
from scipy import cos, sin

from cxroots import Rectangle

class TestCountRoots(unittest.TestCase):
	def setUp(self):
		"""
		Example from "Locating all the Zeros of an Analytic Function in one Complex Variable"
		M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2

		There should be 424 roots inside this contour
		"""
		self.C  = Rectangle([-20.3,20.7], [-20.3,20.7])
		self.f  = lambda z: z**50 + z**12 - 5*sin(20*z)*cos(12*z) - 1
		self.df = lambda z: 50*z**49 + 12*z**11 + 60*sin(12*z)*sin(20*z) - 100*cos(12*z)*cos(20*z)

	def test_countRoots_fdf(self):
		self.assertEqual(self.C.count_roots(self.f, self.df), 424)

	def test_countRoots_f(self):
		self.assertEqual(self.C.count_roots(self.f), 424)

if __name__ == '__main__':
	unittest.main()
