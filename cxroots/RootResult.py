from __future__ import division
import numpy as np

class RootResult(object):
	"""
	A container to hold roots and multiplicities.

	Attributes
	----------
	roots : list
		List of roots
	multiplicities : list
		List of multiplicities where the ith element of the list
		is the multiplicity of the ith element of roots.
	originalContour : Contour
		The contour bounding the region in which the roots
		were found.
	"""
	def __init__(self, roots, multiplicities, originalContour):
		self.roots = roots
		self.multiplicities = multiplicities
		self.originalContour = originalContour

	def show(self, saveFile=None):
		"""
		Plot the roots on the complex plane.

		Parameters
		----------
		saveFile : str, optional
			If provided the plot of the roots will be saved with
			file name saveFile instead of being shown.

		Example
		-------
		.. plot::
			:include-source:
			
			from cxroots import Circle
			C = Circle(0, 2)
			f = lambda z: z**6 + z**3
			r = C.roots(f)
			r.show()
		"""
		import matplotlib.pyplot as plt
		self.originalContour.plot(linecolor='k', linestyle='--')
		plt.scatter(np.real(self.roots), np.imag(self.roots), color='k', marker='x')

		if saveFile is not None:
			plt.savefig(saveFile)
			plt.close()
		else:
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
