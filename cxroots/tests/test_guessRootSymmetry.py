import unittest
from numpy import exp, sin, cos

from cxroots import Circle
from cxroots.CxDerivative import multiplicity_correct
from cxroots.tests.ApproxEqual import roots_approx_equal

class TestGuessRootSymmetry(unittest.TestCase):
	def setUp(self):
		self.C = Circle(0, 3)
		self.f = lambda z: z**4 + z**3 + z**2 + z

		self.roots = [0,-1,1j,-1j]
		self.multiplicities = [1,1,1,1]

	def test_guessRootSymmetry_1(self):
		symmetry = lambda z: [z.conjugate()]
		roots_approx_equal(self.C.roots(self.f, guessRootSymmetry=symmetry), (self.roots, self.multiplicities), decimal=12)

	def test_guessRootSymmetry_2(self):
		# wrong symmetry
		symmetry = lambda z: [z+1]
		roots_approx_equal(self.C.roots(self.f, guessRootSymmetry=symmetry), (self.roots, self.multiplicities), decimal=12)


if __name__ == '__main__':
	unittest.main()
