from __future__ import division
import numpy as np
from collections import deque
import warnings

from .IterativeMethods import iterateToRoot
from .CountRoots import RootError
from .RootResult import RootResult
from .Derivative import find_multiplicity
from .Misc import doc_tab_to_space, docstrings, NumberOfRootsChanged

class MultiplicityError(RuntimeError):
	pass

class countCalls:
	"""
	Count how many times a given function is called.
	"""
	def __init__(self, func):
		self.func = func
		self.calls = 0
		self.points = 0

	def __call__(self, z):
		self.calls += 1
		if hasattr(z, '__len__'):
			self.points += len(z)
		else:
			self.points += 1
		return self.func(z)

@docstrings.get_sectionsf('find_roots_gen')
@docstrings.dedent
@doc_tab_to_space
def find_roots_gen(originalContour, f, df=None, guessRoots=[], guessRootSymmetry=None, 
	newtonStepTol=1e-14, attemptIterBest=True, newtonMaxIter=50, rootErrTol=1e-10, 
	absTol=0, relTol=1e-12, integerTol=0.1, NintAbsTol=0.07, M=5, errStop=1e-8, 
	intMethod='quad', divMax=15, divMin=3, m=2, verbose=False):
	"""
	A generator which at each step takes a contour and either finds all 
	the zeros of f within it or subdivides it further.  Based on the 
	algorithm in [KB].

	Parameters
	----------
	originalContour : :class:`Contour <cxroots.Contour.Contour>`
		The contour which bounds the region in which all the roots of 
		f(z) are sought.
	f : function
		A function of a single complex variable, z, which is analytic 
		within the contour and has no poles or roots on the contour.
	df : function, optional
		A function of a single complex variable which is the derivative 
		of the function f(z). If df is not given then it will be 
		approximated with a finite difference formula.
	guessRoots : list, optional
		A list of known roots or, if the multiplicity is known, a list 
		of (root, multiplicity) tuples.
	guessRootSymmetry : function, optional
		A function of a single complex variable, z, which returns a list
		of all points which are expected to be roots of f, given that z 
		is a root of f.
	newtonStepTol : float, optional
		The required accuracy of the root.  The iterative method used to
		give a final value for each root will exit if the step size, dx,
		between sucessive iterations satisfies abs(dx) < newtonStepTol 
		and iterBestAttempt is False.
	attemptIterBest : bool, optional
		If True then the iterative method used to refine the roots will 
		exit when error of the previous iteration, x0, was at least as 
		good as the current iteration, x, in the sense that 
		abs(f(x)) >= abs(f(x0)) and the previous iteration satisfied 
		abs(dx0) < newtonStepTol.  In this case the preivous iteration 
		is returned as the approximation of the root.
	newtonMaxIter : int, optional
		The iterative method used to give a final value for each root 
		will exit if the number of iterations exceeds newtonMaxIter.
	rootErrTol : float, optional
		A complex value z is considered a root if abs(f(z)) < rootErrTol
	absTol : float, optional
		Absolute error tolerance used by the contour integration.
	relTol : float, optional
		Relative error tolerance used by the contour integration.
	integerTol : float, optional
		A number is considered an integer if it is within integerTol of 
		an integer.  Used when determing if the value for the number of 
		roots within a contour and the values of the computed 
		multiplicities of roots are acceptably close to integers.
	NintAbsTol : float, optional
		The absolute error tolerance used for the contour integration 
		when determining the number of roots within a contour.  Since 
		the result of this integration must be an integer it can be much
		less accurate than usual.
	M : int, optional
		If the number of roots (including multiplicites) within a 
		contour is greater than M then the contour is subdivided
		further.  M must be greater than or equal to the largest 
		multiplcity of any root.  
	errStop : float, optional
		The number of distinct roots within a contour, n, is determined 
		by checking if all the elements of a list of contour integrals 
		involving formal orthogonal polynomials are sufficently close to
		zero, ie. that the absolute value of each element is < errStop.
	intMethod : {'quad', 'romb'}, optional
		If 'quad' then :func:`scipy.integrate.quad` is used to perform the 
		integral.  If 'romb' then Romberg integraion, using 
		:func:`scipy.integrate.romb`, is performed instead.  Typically, quad is 
		the better choice but it requires that the real and imaginary 
		parts of each integral are calculated sepeartely, in addition, 
		if df is not provided, 'quad' will require more function 
		evaluations to approximate df at each point that f is evaluated 
		at.  If evaluating f is expensive then 'romb' may be more 
		efficient since it computes the real and imaginary parts 
		simultaniously and if df is not provided it will approximate it 
		using only the values of f that would be required by the 
		integration routine in any case.
	divMax : int, optional
		If the Romberg integration method is used then divMax is the
		maximum number of divisions before the Romberg integration
		routine exits.
	divMin : int, optional
		If the Romberg integration method is used then divMin is the
		minimum number of divisions before the Romberg integration
		routine is allowed to exit.
	m : int, optional
		Only used if df=None and method='quad'.  The argument order=m is 
		passed to numdifftools.Derivative and is the order of the error 
		term in the Taylor approximation.  m must be even.
	verbose : bool, optional
		If True certain messages concerning the rootfinding process will
		be printed.

	Yields
	------
	list
		Roots of f(z) within the contour originalContour
	list
		Multiplicites of roots
	deque
		The contours which still contain roots
	int
		Remaining number of roots to be found within the contour

	References
	----------
	[KB] Peter Kravanja, Marc Van Barel, "Computing the Zeros of 
	Anayltic Functions", Springer (2000)
	"""	
	# wrap f to record the number of function calls
	f = countCalls(f)

	countKwargs = {'f':f, 'df':df, 'NintAbsTol':NintAbsTol, 'integerTol':integerTol, 
		'divMin':divMin, 'divMax':divMax, 'm':m, 'intMethod':intMethod, 'verbose':verbose}

	try:
		# compute the total number of zeros, including multiplicities, within the originally given contour
		originalContour._numberOfRoots = originalContour.count_roots(**countKwargs)
	except RuntimeError:
		raise RuntimeError("""
			Integration along the initial contour has failed.  
			There is likely a root on or close to the initial contour.
			Try changing the initial contour, if possible.""")

	if verbose:
		print('Total number of roots (counting multiplicities) within the original contour =', originalContour._numberOfRoots)

	roots = []
	multiplicities = []
	failedContours = []
	contours = deque()
	contours.append(originalContour)
	contour = originalContour

	def subdivide(parentContour, NintAbsTol):
		"""
		Given a contour, parentContour, subdivide it into multiple contours.
		"""
		if verbose:
			print('Subdividing', parentContour)

		numberOfRoots = None
		for subcontours in parentContour.subdivisions():
			# if a contour has already been used and caused an error then skip it
			failedContourDesc = list(map(str, failedContours))
			if np.any([contourDesc in failedContourDesc for contourDesc in list(map(str, subcontours))]):
				continue

			# if a contour is near to a known root then skip it
			for root in roots:
				if np.any(np.abs([contour.distance(root) for contour in subcontours]) < 0.01):
					continue

			try:
				numberOfRoots = [contour.count_roots(**countKwargs) for contour in np.array(subcontours)]
				while parentContour._numberOfRoots != sum(numberOfRoots):
					if verbose:
						print('Number of roots in sub contours not adding up to parent contour.')
						print('Recomputing number of roots in parent and child contours with NintAbsTol = ', .5*NintAbsTol)

					tempCountKwargs = countKwargs.copy()
					tempCountKwargs['NintAbsTol'] *= 0.5
					parentContour._numberOfRoots = parentContour.count_roots(**tempCountKwargs)
					numberOfRoots = [contour.count_roots(**tempCountKwargs) for contour in np.array(subcontours)]

				if parentContour._numberOfRoots == sum(numberOfRoots):
					break

			except RootError:
				# If the number of zeros within either of the new contours is not an integer then it is
				# likely that the introduced line which subdivides 'parentContour' lies on a zero.
				# To avoid this we will try to place the subdividing line at a different point along 
				# the division axis
				if verbose:
					print('RootError encountered when subdivding', parentContour, 'into:')
					print(subcontours[0])
					print(subcontours[1])
				continue

		if numberOfRoots is None or parentContour._numberOfRoots != sum(numberOfRoots):
			# The list of subdivisions has been exhaused and still the number of enclosed zeros does not add up 
			raise RuntimeError("""Unable to subdivide contour:
				\t%s
				""" % parentContour)

		# record number of roots within each sub-contour
		for i, contour in enumerate(subcontours):
			contour._numberOfRoots = numberOfRoots[i]

		# add these sub-contours to the list of contours to find the roots in
		contours.extend([contour for i, contour in enumerate(subcontours) if numberOfRoots[i] != 0])


	def remove_siblings_children(C):
		"""
		Remove the contour C and all its siblings and children from the 
		list of contours to be examined/subdivided.
		"""

		# remove itself
		try:
			contours.remove(C)
		except ValueError:
			pass

		# get sibling and child contours
		# siblings:
		relations = C._parentContour._childcontours 
		relations.remove(C)

		# children:
		if hasattr(C, '_childcontours'):
			relations.extend(C._childcontours)

		# interate over all relations
		for relation in relations:
			remove_relations(relation)


	def addRoot(root, multiplicity=None):
		# check that the root we have found is distinct from the ones we already have
		if not roots or np.all(abs(np.array(roots) - root) > newtonStepTol):
			# add the root to the list if it is within the original contour
			if originalContour.contains(root):
				roots.append(root)
				if multiplicity is None:
					multiplicity = find_multiplicity(f, root, df=df, rootErrTol=rootErrTol)

				multiplicities.append(multiplicity.real)

			# check to see if there are any other roots implied by the given symmetry
			if guessRootSymmetry is not None:
				for x0 in guessRootSymmetry(root):
					# first check that x0 is distinct from the roots we already have
					if np.all(abs(np.array(roots) - x0) > newtonStepTol):
						if verbose:
							print(root, 'is a root so guessRootSymmetry suggests that', x0, 'might also be a root.')
						root = iterateToRoot(x0, f, df, newtonStepTol, rootErrTol, newtonMaxIter, attemptIterBest, verbose)
						if root is not None:
							contours.append(Circle(root, 1e-3))

		elif verbose:
			print('Already recorded root', root)

	# Add contours surrounding known roots so that they will be checked
	for root in guessRoots:
		contours.append(Circle(root, 1e-3))

	while contours:
		# yield the initial state here so that the animation in demo_find_roots shows the first frame
		totFoundRoots = sum(int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities))
		yield roots, multiplicities, contours, originalContour._numberOfRoots - totFoundRoots

		contour = contours.pop()

		if not hasattr(contour, '_numberOfRoots'):
			# occours when testing a guess using a circle centered around the guess root
			contour._numberOfRoots = contour.count_roots(**countKwargs)

		if verbose:
			print(contour._numberOfRoots, 'roots in', contour)

		# if a known root is too near to this contour then reverse the subdivision that created it 
		if np.any([contour.distance(root) < newtonStepTol for root in roots]):
			# remove the contour together with its children and siblings
			remove_siblings_children(contour)

			# put the parent contour back into the list of contours to be subdivided again
			contours.append(contour._parentContour)

			# do not use this contour again
			failedContours.append(contour)
			continue

		# if the contour is smaller than the newtonStepTol then just assume that 
		# the root is at the center of the contour, print a warning and move on
		if contour.area < newtonStepTol:
			warnings.warn("""The area of the interior of this contour with center %i is smaller than newtonStepTol!
				The center of the contour has been recored as a root of multiplicity %i but this could not be verified.
				rootErrTol may be too small."""%(contour.centralPoint, contour._numberOfRoots))
			addRoot(root=contour.centralPoint,  multiplicity=contour._numberOfRoots)
			continue

		# if all the roots within the contour have been located then continue to the next contour
		numberOfKnownRootsInContour = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if contour.contains(root)])
		if contour._numberOfRoots == numberOfKnownRootsInContour:
			continue

		# if there are too many roots within the contour then subdivide or
		# if there are any known roots within the contour then also subdivide 
		# (so as not to waste time re-approximating these roots)
		if numberOfKnownRootsInContour > 0 or contour._numberOfRoots > M:
			subdivide(contour, NintAbsTol)

		else:
			# Approximate the roots in this contour
			try:
				approxRoots, approxRootMultiplicities = contour.approximate_roots(contour._numberOfRoots, f, df, 
					absTol=absTol, relTol=relTol, integerTol=integerTol, errStop=errStop, divMin=divMin, divMax=divMax, 
					m=m, rootTol=newtonStepTol, intMethod=intMethod, verbose=verbose)
			except (MultiplicityError, NumberOfRootsChanged):
				subdivide(contour, NintAbsTol)
				continue

			for approxRoot, approxRootMultiplicity in list(zip(approxRoots, approxRootMultiplicities)):
				# attempt to refine the root
				root = iterateToRoot(approxRoot, f, df, newtonStepTol, rootErrTol, newtonMaxIter, attemptIterBest, verbose)

				if abs(f(approxRoot)) < rootErrTol and (root is None or abs(f(approxRoot)) < abs(f(root))):
					# stick with the original approximation
					root = approxRoot

				if root is not None:
					# if the root turns out to be very close to the contour then this may have
					# introduced an error.  Therefore, compute the multiplicity of this root
					# directly and disregard this contour (repeat its  parent's subdivision).
					if contour.distance(root) < newtonStepTol:
						# remove the contour and any relations
						remove_siblings_children(contour)

						# put the parent contour back into the list of contours to subdivide again
						parent = contour._parentContour
						contours.append(parent)

						# compute the root multiplicity directly
						approxRootMultiplicity = None

						# do not use this contour again
						failedContours.append(contour)
					
					# if we found a root add it to the list of known roots
					addRoot(root, approxRootMultiplicity)

			# if we haven't found all the roots then subdivide further
			numberOfKnownRootsInContour = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if contour.contains(root)])
			if contour._numberOfRoots != numberOfKnownRootsInContour and contour not in failedContours:
				subdivide(contour, NintAbsTol)


	# delete cache for original contour incase this contour is being reused
	for segment in originalContour.segments:
		segment._integralCache = {}
		segment._trapValuesCache = {}

	if verbose:
		print('Completed rootfinding with', f.calls, 'evaluations of f at', f.points, 'points')

	# yield one last time so that the animation shows the final frame
	yield roots, multiplicities, contours, originalContour._numberOfRoots - totFoundRoots

@docstrings.dedent
@doc_tab_to_space
def find_roots(originalContour, f, df=None, **kwargs):
	"""
	Find all the roots of the complex analytic function f within the 
	given contour.

	Parameters
	----------
	%(find_roots_gen.parameters)s

	Returns
	-------
	result : :class:`RootResult <cxroots.RootResult.RootResult>`
		A container for the roots and their multiplicities.
	"""
	rootFinder = find_roots_gen(originalContour, f, df, **kwargs)
	for roots, multiplicities, contours, numberOfRemainingRoots in rootFinder:
		pass
	return RootResult(roots, multiplicities, originalContour)

