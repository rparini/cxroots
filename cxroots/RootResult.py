from __future__ import division
import numpy as np

class RootResult(object):
	"""
	An object to hold the results of rootfinding
	"""
	def __init__(self, roots, multiplicities, originalContour):
		self.roots = roots
		self.multiplicities = multiplicities
		self.originalContour = originalContour

	def show(self):
		"""
		Plot the roots on the complex plane
		"""
		import matplotlib.pyplot as plt
		self.originalContour.plot(linecolor='k', linestyle='--')
		plt.scatter(np.real(self.roots), np.imag(self.roots), color='k', marker='x')
		plt.show()

	def __str__(self):
		roots, multiplicities = np.array(self.roots), np.array(self.multiplicities)

		# reorder roots
		sortargs = np.argsort(roots)
		roots, multiplicities = roots[sortargs], multiplicities[sortargs]

		s =  ' Multiplicity |               Root              '
		s+='\n------------------------------------------------'

		for i, root in np.ndenumerate(roots):
			if root.real < 0:
				s += '\n{: ^14d}| {:.12f} {:+.12f}i'.format(int(multiplicities[i]), root.real, root.imag)
			else:
				s += '\n{: ^14d}|  {:.12f} {:+.12f}i'.format(int(multiplicities[i]), root.real, root.imag)

		return s
