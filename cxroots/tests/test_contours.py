import pytest

from cxroots import AnnulusSector

def test_PolarRect_contains():
	r0=8.938
	r1=9.625
	phi0=6.126
	phi1=6.519

	z = (9-1.04825594683e-18j)
	C = AnnulusSector(0, [r0,r1], [phi0,phi1])

	# import matplotlib.pyplot as plt
	# plt.scatter((9),(0))
	# C.show()

	assert C.contains(z)

