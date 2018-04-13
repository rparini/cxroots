from __future__ import division
import random
import warnings
import numpy as np
import scipy.integrate
import scipy.linalg
from scipy import pi, exp, sin, log
import scipy

class ComplexPath(object):
	""" A base class for paths in the complex plane """
	def __init__(self):
		self._integralCache = {}
		self._contArgCache = {}
		self._trapValuesCache = {}

	def trapValues(self, f, k, useCache=True):
		"""
		2**k+1 is the number of required points for the function f to
		be evaluated at.
		"""

		if f in self._trapValuesCache.keys() and useCache:
			vals = self._trapValuesCache[f]
			vals_k = int(np.log2(len(vals)-1))
			
			if vals_k == k:
				return vals
			elif vals_k > k:
				return vals[::2**(vals_k-k)]
			else:
				t = np.linspace(0, 1, 2**k+1)
				vals = np.empty(2**k+1, dtype=np.complex128)
				vals.fill(np.nan)
				vals[::2**(k-vals_k)] = self._trapValuesCache[f]
				vals[np.isnan(vals)] = f(self(t[np.isnan(vals)]))

				# cache values
				self._trapValuesCache[f] = vals
				return vals

		else:
			t = np.linspace(0, 1, 2**k+1)
			vals = f(self(t))
			if useCache:
				self._trapValuesCache[f] = vals
			return vals


	def plot(self, N=100, linecolor='C0', linestyle='-'):
		"""
		Use matplotlib to plot, but not show, the path as a 
		2D plot in the Complex plane.  To show use pyplot.show()

		Parameters
		----------
		N : int, optional
			The number of points to use when plotting the path.  Default is 100
		linecolor : optional
			The colour of the plotted path, passed to the pyplot.plot function 
			as the keyword argument of 'color'.
		linestyle : str, optional
			The line style of the plotted path, passed to the pyplot.plot 
			function as the keyword argument of 'linestyle'.  Default is '-' 
			which corresponds to a solid line.
		"""
		import matplotlib.pyplot as plt
		t = np.linspace(0,1,N)
		path = self(t)
		plt.plot(path.real, path.imag, color=linecolor, linestyle=linestyle)
		plt.xlabel('Re[$z$]', size=16)
		plt.ylabel('Im[$z$]', size=16)
		plt.gca().set_aspect(1)
		plt.tight_layout()

		# add arrow to indicate direction of path
		arrow_direction = (self(0.51) - self(0.5))/abs(self(0.51) - self(0.5))
		arrow_extent = 1e-6*arrow_direction
		ymin, ymax = plt.gca().get_ylim()
		xmin, xmax = plt.gca().get_xlim()
		head_length = max(abs(ymax - ymin), abs(xmax - xmin))/40.
		plt.arrow(self(0.5).real, self(0.5).imag,
				  arrow_extent.real, arrow_extent.imag,
				  head_width=head_length*2/3., head_length=head_length, 
				  fc=linecolor, ec=linecolor)

	def show(self, *args, **kwargs):
		""" Shows the path as a 2D plot in the complex plane using 
		the same arguments as the plot method """
		import matplotlib.pyplot as plt
		self.plot(*args, **kwargs)
		plt.show()

	def integrate(self, f, absTol=0, relTol=1e-12, rombergDivMax=10, method='quad', show=False):
		"""
		Integrate the function f along the path using SciPy's Romberg
		algorithm.  The value of the integral is cached and will be
		reused if the method is called with same function f and tol. 

		Parameters
		----------
		f : function of a single complex variable
		tol : float, optional
			The absolute tolerance passed to SciPy's Romberg function.
			Default is 1e-8.
		rombergDivMax : int, optional
			The maximum order of extrapolation passed to SciPy's Romberg function

		Returns
		-------
		integral : complex
			The integral of the function f along the path.
		"""

		args = (f, absTol, relTol, rombergDivMax, method)
		if args in self._integralCache.keys():
			integral = self._integralCache[args]

		elif hasattr(self, '_reversePath') and args in self._reversePath._integralCache:
			# if we have already computed the reverse of this path
			integral = -self._reversePath._integralCache[args]

		else:			
			# suppress accuracy warnings
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				integrand = lambda t: f(self(t))*self.dzdt(t)

				if method == 'romb':
					integral = scipy.integrate.romberg(integrand, 0, 1, tol=absTol, rtol=relTol, divmax=rombergDivMax, show=show)
				elif method == 'quad':
					integrand_real = lambda t: np.real(integrand(t))
					integrand_imag = lambda t: np.imag(integrand(t))

					integral_real, abserr_real = scipy.integrate.quad(integrand_real, 0, 1, epsabs=absTol, epsrel=relTol)
					integral_imag, abserr_imag = scipy.integrate.quad(integrand_imag, 0, 1, epsabs=absTol, epsrel=relTol)
					integral = integral_real + 1j*integral_imag

			if np.isnan(integral):
				raise RuntimeError('The integral along the segment %s is NaN.\
					\nThis is most likely due to a root being on or very close to the path of integration.'%self)

			self._integralCache[args] = integral

		return integral


class ComplexLine(ComplexPath):
	""" A straight line in the complex plane from a to b """
	def __init__(self, a, b):
		self.a, self.b = a, b
		self.dzdt = lambda t: self.b-self.a
		super(ComplexLine, self).__init__()

	def __str__(self):
		return 'ComplexLine from %.3f+%.3fi to %.3f+%.3fi' % (self.a.real, self.a.imag, self.b.real, self.b.imag)

	def __call__(self, t):
		""" The parameterization of the line in the variable t, where 0 <= t <= 1 """
		return self.a + t*(self.b-self.a)

	def distance(self, P):
		""" Distance from the point P to the closest point on the line. """
		a, b = self.a, self.b
		return abs((b.real-a.real)*(a.imag-P.imag)-(b.imag-a.imag)*(a.real-P.real))/abs(b-a)


class ComplexArc(ComplexPath):
	""" An arc with center z0, radius R, initial angle t0 and change of angle dt """
	def __init__(self, z0, R, t0, dt):
		self.z0, self.R, self.t0, self.dt = z0, R, t0, dt
		self.dzdt = lambda t: 1j*self.dt*self.R*exp(1j*(self.t0 + t*self.dt))
		super(ComplexArc, self).__init__()

	def __str__(self):
		return 'ComplexArc: z0=%.3f, R=%.3f, t0=%.3f, dt=%.3f' % (self.z0, self.R, self.t0, self.dt)

	def __call__(self, t):
		""" The parameterization of the arc in the variable t, where 0 <= t <= 1 """
		return self.R*exp(1j*(self.t0 + t*self.dt)) + self.z0

	def distance(self, P):
		""" Distance from the point P to the closest point on the arc .
		XXX: Needs to be properly implemented. """
		t = np.linspace(0,1,100000)
		return np.min(np.abs(self(t) - P))
