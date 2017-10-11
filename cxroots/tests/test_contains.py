import unittest

from cxroots import AnnulusSector

class TestContains(unittest.TestCase):
	def test_PolarRect_contains(self):
		r0=8.938
		r1=9.625
		phi0=6.126
		phi1=6.519

		z = (9-1.04825594683e-18j)
		C = AnnulusSector(0, [r0,r1], [phi0,phi1])

		# import matplotlib.pyplot as plt
		# plt.scatter((9),(0))
		# C.show()

		self.assertTrue(C.contains(z))

if __name__ == '__main__':
	unittest.main(verbosity=3)
