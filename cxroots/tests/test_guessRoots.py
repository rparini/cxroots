import unittest
from numpy import exp, sin, cos

from cxroots import Circle
from cxroots.tests.ApproxEqual import roots_approx_equal

class TestGuessRoots(unittest.TestCase):
	def setUp(self):
		self.C = Circle(0, 3)
		self.f = lambda z: (z-2.5)**2 * (exp(-z)*sin(z/2)-1.2*cos(z))

		self.roots = [2.5,
					  1.44025113016670301345110737, 
					  -0.974651035111059787741822566 - 1.381047768247156339633038236j,
					  -0.974651035111059787741822566 + 1.381047768247156339633038236j]
		self.multiplicities = [2,1,1,1]

	def test_guessRoot(self):
		roots_approx_equal(self.C.roots(self.f, guessRoots=[2.5], verbose=True), (self.roots, self.multiplicities), decimal=12)

	def test_incorrect_guessRoot(self):
		roots_approx_equal(self.C.roots(self.f, guessRoots=[2.5, 3], verbose=True), (self.roots, self.multiplicities), decimal=12)

if __name__ == '__main__':
	unittest.main()
