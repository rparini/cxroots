from __future__ import division

import numpy as np
import scipy.integrate
from numpy import exp, pi

class ComplexPath(object):
	"""A base class for paths in the complex plane."""
	def __init__(self):
		self._integralCache = {}
		self._trapValuesCache = {}


	def __call__(self, t):
		r"""
		The parameterization of the path in the varaible :math:`t\in[0,1]`.

		Parameters
		----------
		t : float
			A real number :math:`0\leq t \leq 1`.

		Returns
		-------
		complex
			A point on the path in the complex plane.
		"""
		raise NotImplementedError('__call__ must be implemented in a subclass')


	def trap_values(self, f, k, useCache=True):
		"""
		Compute or retrieve (if cached) the values of the functions f
		at :math:`2^k+1` points along the contour which are evenly
		spaced with respect to the parameterisation of the contour.

		Parameters
		----------
		f : function
			A function of a single complex variable.
		k : int
			Defines the number of points along the curve that f is to be
			evaluated at as :math:`2^k+1`.
		useCache : bool, optional
			If True then use, if available, the results of any previous
			calls to this function for the same f and save any new
			results so that they can be reused later.

		Returns
		-------
		:class:`numpy.ndarray`
			The values of f at :math:`2^k+1` points along the contour
			which are evenly spaced with respect to the parameterisation
			of the contour.
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
		Uses matplotlib to plot, but not show, the path as a 2D plot in
		the Complex plane.

		Parameters
		----------
		N : int, optional
			The number of points to use when plotting the path.
		linecolor : optional
			The colour of the plotted path, passed to the
			:func:`matplotlib.pyplot.plot` function as the keyword
			argument of 'color'.  See the matplotlib tutorial on
			`specifying colours <https://matplotlib.org/users/colors.html#>`_.
		linestyle : str, optional
			The line style of the plotted path, passed to the
			:func:`matplotlib.pyplot.plot` function as the keyword
			argument of 'linestyle'.  The default corresponds to a solid
			line.  See :meth:`matplotlib.lines.Line2D.set_linestyle` for
			other acceptable arguments.
		"""
		import matplotlib.pyplot as plt
		t = np.linspace(0,1,N)
		path = self(t)
		plt.plot(path.real, path.imag, color=linecolor, linestyle=linestyle)
		plt.xlabel('Re[$z$]', size=16)
		plt.ylabel('Im[$z$]', size=16)
		plt.gca().set_aspect(1)

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

	def show(self, saveFile=None, **plotKwargs):
		"""
		Shows the path as a 2D plot in the complex plane.  Requires
		Matplotlib.

		Parameters
		----------
		saveFile : str (optional)
			If given then the plot will be saved to disk with name
			'saveFile'.  If saveFile=None the plot is shown on-screen.
		**plotKwargs
			Other key word arguments are passed to :meth:`~cxroots.Paths.ComplexPath.plot`.
		"""
		import matplotlib.pyplot as plt
		self.plot(**plotKwargs)

		if saveFile is not None:
			plt.savefig(saveFile, bbox_inches='tight')
			plt.close()
		else:
			plt.show()

	def integrate(self, f, absTol=0, relTol=1e-12, divMax=15, intMethod='quad', verbose=False):
		"""
		Integrate the function f along the path.  The value of the
		integral is cached and will be reused if the method is called
		with same arguments (ignoring verbose).

		Parameters
		----------
		f : function
			A function of a single complex variable.
		absTol : float, optional
			The absolute tolerance for the integration.
		relTol : float, optional
			The realative tolerance for the integration.
		divMax : int, optional
			If the Romberg integration method is used then divMax is the
			maximum number of divisions before the Romberg integration
			routine of a path exits.
		intMethod : {'quad', 'romb'}, optional
			If 'quad' then :func:`scipy.integrate.quad` is used to
			compute the integral.  If 'romb' then Romberg integraion,
			using :func:`scipy.integrate.romberg`, is used instead.
		verbose : bool, optional
			Passed ass the `show` argument of :func:`scipy.integrate.romberg`.

		Returns
		-------
		complex
			The integral of the function f along the path.

		Notes
		-----
		This function is only used when checking the
		multiplicity of roots.  The bulk of the integration for
		rootfinding is done with :func:`cxroots.CountRoots.prod`.
		"""

		args = (f, absTol, relTol, divMax, intMethod)
		if args in self._integralCache.keys():
			integral = self._integralCache[args]

		elif hasattr(self, '_reversePath') and args in self._reversePath._integralCache:
			# if we have already computed the reverse of this path
			integral = -self._reversePath._integralCache[args]

		else:
			integrand = lambda t: f(self(t))*self.dzdt(t)

			if intMethod == 'romb':
				integral = scipy.integrate.romberg(integrand, 0, 1, tol=absTol, rtol=relTol, divmax=divMax, show=verbose)
			elif intMethod == 'quad':
				integrand_real = lambda t: np.real(integrand(t))
				integrand_imag = lambda t: np.imag(integrand(t))

				integral_real, abserr_real = scipy.integrate.quad(integrand_real, 0, 1, epsabs=absTol, epsrel=relTol)
				integral_imag, abserr_imag = scipy.integrate.quad(integrand_imag, 0, 1, epsabs=absTol, epsrel=relTol)
				integral = integral_real + 1j*integral_imag
			else:
				raise ValueError("intMethod must be either 'romb' or 'quad'")

			if np.isnan(integral):
				raise RuntimeError('The integral along the segment %s is NaN.\
					\nThis is most likely due to a root being on or very close to the path of integration.'%self)

			self._integralCache[args] = integral

		return integral


class ComplexLine(ComplexPath):
	r"""
	A straight line :math:`z` in the complex plane from a to b
	parameterised by

	..math::

		z(t) = a + (b-a)t, \quad 0\leq t \leq 1


	Parameters
	----------
	a : float
	b : float
	"""
	def __init__(self, a, b):
		self.a, self.b = a, b
		self.dzdt = lambda t: self.b-self.a
		super(ComplexLine, self).__init__()

	def __str__(self):
		return 'ComplexLine from %.3f+%.3fi to %.3f+%.3fi' % (self.a.real, self.a.imag, self.b.real, self.b.imag)

	def __call__(self, t):
		r"""
		The function :math:`z(t) = a + (b-a)t`.

		Parameters
		----------
		t : float
			A real number :math:`0\leq t \leq 1`.

		Returns
		-------
		complex
			A point on the line in the complex plane.
		"""
		return self.a + t*(self.b-self.a)

	def distance(self, z):
		"""
		Distance from the point z to the closest point on the line.

		Parameters
		----------
		z : complex

		Returns
		-------
		float
			The distance from z to the point on the line which is
			closest to z.
		"""
		# convert complex numbers to vectors
		A = np.array([self.a.real, self.a.imag])
		B = np.array([self.b.real, self.b.imag])
		Z = np.array([z.real, z.imag])

		# the projection of the point z onto the line a -> b is where
		# the parameter t is
		t = (Z-A).dot(B-A)/abs((B-A).dot(B-A))

		# but the line segment only has 0 <= t <= 1
		t = t.clip(0,1)

		# so the point on the line segment closest to z is
		c = self(t)
		return abs(c-z)

class ComplexArc(ComplexPath):
	r"""
	A circular arc :math:`z` with center z0, radius R, initial angle t0
	and change of angle dt.  The arc is parameterised by

	..math::

		z(t) = R e^{i(t0 + t dt)} + z0, \quad 0\leq t \leq 1

	Parameters
	----------
	z0 : complex
	R : float
	t0 : float
	dt : float
	"""
	def __init__(self, z0, R, t0, dt):
		self.z0, self.R, self.t0, self.dt = z0, R, t0, dt
		self.dzdt = lambda t: 1j*self.dt*self.R*exp(1j*(self.t0 + t*self.dt))
		super(ComplexArc, self).__init__()

	def __str__(self):
		return 'ComplexArc: z0=%.3f, R=%.3f, t0=%.3f, dt=%.3f' % (self.z0, self.R, self.t0, self.dt)

	def __call__(self, t):
		r"""
		The function :math:`z(t) = R e^{i(t_0 + t dt)} + z_0`.

		Parameters
		----------
		t : float
			A real number :math:`0\leq t \leq 1`.

		Returns
		-------
		complex
			A point on the arc in the complex plane.
		"""
		return self.R*exp(1j*(self.t0 + t*self.dt)) + self.z0

	def distance(self, z):
		"""
		Distance from the point z to the closest point on the arc.

		Parameters
		----------
		z : complex

		Returns
		-------
		float
			The distance from z to the point on the arc which is closest
			to z.
		"""
		theta = np.angle(z-self.z0) 				# np.angle maps to (-pi,pi]
		theta = (theta-self.t0)%(2*pi) + self.t0 	# put theta in [t0,t0+2pi)

		if ((self.dt > 0 and self.t0 < theta < self.t0+self.dt)
			or (self.dt < 0 and self.t0+self.dt < theta - 2*pi < self.t0)):
			# the closest point to z lies on the arc
			return abs(self.R*exp(1j*theta) + self.z0 - z)
		else:
			# the closest point to z is one of the endpoints
			return min(abs(self(0)-z), abs(self(1)-z))

