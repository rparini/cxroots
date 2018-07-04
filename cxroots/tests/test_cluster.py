import unittest
import pytest
import numpy as np

from cxroots import Circle, Rectangle
from cxroots.tests.ApproxEqual import roots_approx_equal

class ClusterTest(object):
	def test_rootfinding_df(self):
		roots_approx_equal(self.C.roots(self.f, self.df, verbose=True), (self.roots, self.multiplicities), decimal=12)

	def test_rootfinding_f(self):
		roots_approx_equal(self.C.roots(self.f), (self.roots, self.multiplicities), decimal=12)

class TestCluster_1_rect(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.01, 3.02, 8, 8.02, 8+0.01j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Rectangle([2,9], [-1,1])
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

class TestCluster_1_circle(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.01, 3.02, 8, 8.02, 8+0.01j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Circle(0, 8.5)
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

class TestCluster_2_rect(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.001, 3.002, 8, 8.002, 8+0.001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Rectangle([2,9], [-1,1])
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

class TestCluster_2_circle(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.001, 3.002, 8, 8.002, 8+0.001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Circle(0, 8.5)
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

class TestCluster_3_rect(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.0001, 3.0002, 8, 8.0002, 8+0.0001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Rectangle([2,9], [-1,1])
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

class TestCluster_3_circle(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.0001, 3.0002, 8, 8.0002, 8+0.0001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Circle(0, 8.5)
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

@pytest.mark.xfail(reason='Cluster of roots too tight')
class TestCluster_4_rect(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.00001, 3.00002, 8, 8.00002, 8+0.00001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Rectangle([2,9], [-1,1])
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])

@pytest.mark.xfail(reason='Cluster of roots too tight')
class TestCluster_4_circle(unittest.TestCase, ClusterTest):
	def setUp(self):
		self.roots = roots = [3, 3.00001, 3.00002, 8, 8.00002, 8+0.00001j]
		self.multiplicities = [1,1,1,1,1,1]

		self.C = Circle(0, 8.5)
		self.f = lambda z: np.prod([z-r for r in roots])
		self.df = lambda z: np.sum([np.prod([z-r for r in np.delete(roots,i)]) for i in range(len(roots))])


if __name__ == '__main__':
	unittest.main(verbosity=3)
