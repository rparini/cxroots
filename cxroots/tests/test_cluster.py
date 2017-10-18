import unittest
import numpy as np
from scipy import pi, sqrt, exp, sin, cos

from cxroots import Circle, Rectangle
from cxroots.tests.ApproxEqual import roots_approx_equal

### XXX: Need some way to distinguish clusters of roots

class TestCluster_1(unittest.TestCase):
	def setUp(self):
		self.roots = roots = [3, 3.00001, 3.00002, 8, 8.00002, 8+0.00001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Rectangle([2,9], [-1,1])
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

	def test_rootfinding_df(self):
		roots_approx_equal(self.C.roots(self.f, self.df), (self.roots, self.multiplicities), decimal=12)

	# def test_rootfinding_f(self):
	# 	roots_approx_equal(self.C.roots(self.f), (self.roots, self.multiplicities), decimal=12)

# class TestCluster_2(unittest.TestCase):
# 	def setUp(self):
# 		self.roots = roots = [3, 3.00001, 3.00002, 8, 8.00002, 8+0.00001j]
# 		self.multiplicities = [1,1,1,1,1,1]

# 		self.C = Circle(0, 8.5)
# 		self.f = lambda z: np.prod([z-r for r in roots])
# 		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

# 	def test_rootfinding_df(self):
# 		roots_approx_equal(self.C.roots(self.f, self.df), (self.roots, self.multiplicities), decimal=12)

# 	def test_rootfinding_f(self):
# 		roots_approx_equal(self.C.roots(self.f), (self.roots, self.multiplicities), decimal=12)


if __name__ == '__main__':
	# unittest.main(verbosity=3)

	roots = [3, 3.00001, 3.00002, 8, 8.00002, 8+0.00001j]
	multiplicities = [1,1,1,1,1,1]

	f = lambda z: np.prod([z-r for r in roots])
	df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

	C = Circle(0, 8.5)
	# C = Rectangle([2,9], [-1,1])

	# C.approximate_roots(f, df, verbose=True)
	C.demo_roots(f, df, verbose=True)
