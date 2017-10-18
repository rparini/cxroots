import unittest
import numpy as np

from cxroots.IterativeMethods import muller
from scipy import pi, cos, sin

class TestSecant(unittest.TestCase):
	def test_secant(self):
		# example from Table 2.5 of "Numerical Analysis" by Richard L. Burden, J. Douglas Faires
		f  = lambda x: cos(x)-x**2 + x**3 * 1j

		iterations = []
		callback = lambda x, dx, y, iteration: iterations.append(x)
		x, err = muller(0.5, pi/4, 0.6, f)

		root = 0.7296100078977741539157356847 + 0.1570923181734581909733787621j
		self.assertAlmostEqual(x, root)

if __name__ == '__main__':
	unittest.main()
