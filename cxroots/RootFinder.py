from __future__ import division
import numpy as np
import cmath
from scipy import pi, exp, sin
import scipy
from numpy.random import uniform
import itertools
from collections import deque
import warnings

from .IterativeMethods import iterateToRoot
from .CountRoots import prod, RootError
from .RootResult import RootResult
from .CxDerivative import multiplicity_correct

from .Misc import doc_tab_to_space, docstrings

class MultiplicityError(RuntimeError):
	pass

@docstrings.get_sectionsf('findRootsGen')
@docstrings.dedent
@doc_tab_to_space
def findRootsGen(originalContour, f, df=None, guessRoots=[], guessRootSymmetry=None, 
	newtonStepTol=1e-14, attemptIterBest=True, newtonMaxIter=50, rootErrTol=1e-10, 
	absTol=0, relTol=1e-12, integerTol=0.1, NintAbsTol=0.07, M=5, errStop=1e-8, 
	intMethod='quad', divMax=15, divMin=5, m=2, verbose=False):
	"""
	A generator which at each step takes a contour and either finds 
	all the zeros of f within it or subdivides it further.  Based
	on the algorithm in [KB]

	Parameters
	----------
	originalContour : Contour
		The contour C which bounds the region in which all the 
		roots of f(z) are sought.
	f : function
		A function of a single complex variable, z, which is 
		analytic within C and has no poles or roots on C.
	df : function, optional
		A function of a single complex variable which is the 
		derivative of the function f(z). If df is not given 
		then it will be approximated with a finite difference 
		formula.
	guessRoots : list, optional
		A list of known roots or, if the multiplicity is known, 
		a list of (root, multiplicity) tuples.
	guessRootSymmetry : function, optional
		A function of a single complex variable, z, which 
		returns a list of all points which are expected to be 
		roots of f, given that z is a root of f.
	newtonStepTol : float, optional
		The required accuracy of the root.
		The iterative method used to give a final value for each
		root will exit if the step size, dx, between sucessive 
		iterations satisfies abs(dx) < newtonStepTol and iterBestAttempt
		is False.
	attemptIterBest : bool, optional
		If True then the iterative method used to refine the roots will exit 
		when error of the previous iteration, x0, was at least as good as the 
		current iteration, x, in the sense that abs(f(x)) >= abs(f(x0)) and 
		the previous iteration satisfied abs(dx0) < newtonStepTol.In this 
		case the preivous iteration is returned as the approximation of the root.
	newtonMaxIter : int, optional
		The iterative method used to give a final value for each
		root will exit if the number of iterations exceeds newtonMaxIter.
	rootErrTol : float, optional
		A complex value z is considered a root if abs(f(z)) < rootErrTol
	absTol : float, optional
		Absolute error tolerance used by the contour integration.
	relTol : float, optional
		Relative error tolerance used by the contour integration.
	integerTol : float, optional
		A number is considered an integer if it is within 
		integerTol of an integer.  Used when determing if the
		value for the number of roots within a contour and the
		values of the computed multiplicities of roots are
		acceptably close to integers.
	NintAbsTol : float, optional
		The absolute error tolerance used for the contour 
		integration when determining the number of roots 
		within a contour.  Since the result of this integration
		must be an integer it can be much less accurate
		than usual.
	M : int, optional
		If the number of roots (including multiplicites) within a
		contour is greater than M then the contour is subdivided
		further.  M must be greater than or equal to the largest 
		multiplcity of any root.  
	errStop : float, optional
		The number of distinct roots within a contour, n, is 
		determined by checking if all the elements of a list of 
		contour integrals involving formal orthogonal polynomials 
		are sufficently close to zero, ie. that the absolute 
		value of each element is < errStop.
	intMethod : str, optional
		Either 'quad' to integrate using scipy.integrate.quad or
		'romb' to integrate using Romberg's method.
	divMax : int, optional
		If the Romberg integration method is used then divMax is the
		maximum number of divisions before the Romberg integration
		routine of a path exits.
	divMin : int, optional
		If the Romberg integration method is used then divMin is the
		minimum number of divisions before the Romberg integration
		routine of a path is allowed to exit.
    m : int, optional
    	Only used if df=None.  If method='romb' then m defines the stencil size for the 
    	numerical differentiation of f, passed to numdifftools.fornberg.fd_derivative.
    	The stencil size is of 2*m+1 points in the interior, and 2*m+2 points for each 
    	of the 2*m boundary points.  If instead method='quad' then m must is the order of 
    	the error term in the Taylor approximation used which must be even.  The argument
    	order=m is passed to numdifftools.Derivative.
	verbose : bool, optional
		If True certain messages concerning the rootfinding process
		will be printed.

	Yields
	------
	list
		Roots of f(z) within the contour C
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
	try:
		# total number of zeros, including multiplicities
		totNumberOfRoots = originalContour.count_roots(f, df, NintAbsTol, integerTol, divMin, divMax, m, intMethod, verbose)
		originalContour._numberOfRoots = totNumberOfRoots
	except RuntimeError:
		raise RuntimeError("""
			Integration along the initial contour has failed.  There is likely a root on or close to the initial contour
			Try changing the initial contour, if possible.""")

	if verbose:
		print('Total number of roots (counting multiplicities) within original contour:', totNumberOfRoots)

	smallBoxWarning = False
	roots = []
	multiplicities = []
	failedBoxes = []
	boxes = deque()
	boxes.append((originalContour,totNumberOfRoots))

	def subdivide(parentBox):
		"""
		Given a contour, parentBox, subdivide it into multiple contours.
		"""
		numberOfRoots = None
		for subBoxes in parentBox.subdivisions():
			# if a box has already been used and caused an error then skip it
			failedBoxDesc = list(map(str, failedBoxes))
			if np.any([boxDesc in failedBoxDesc for boxDesc in list(map(str, subBoxes))]):
				continue

			# XXX: if a root is near to a box then subdivide again?

			try:
				numberOfRoots = [box.count_roots(f, df, NintAbsTol, integerTol, divMin, divMax, m, intMethod, verbose) for box in np.array(subBoxes)]
				if parentBox._numberOfRoots == sum(numberOfRoots):
					break
			except RootError:
				# If the number of zeros within either of the new contours is not an integer then it is
				# likely that the introduced line which subdivides 'parentBox' lies on a zero.
				# To avoid this we will try to place the subdividing line at a different point along 
				# the division axis
				continue

		if numberOfRoots is None or parentBox._numberOfRoots != sum(numberOfRoots):
			# The list of subdivisions has been exhaused and still the number of enclosed zeros does not add up 
			raise RuntimeError("""Unable to subdivide box:
				\t%s
				""" % parentBox)

		boxes.extend([(box, numberOfRoots[i]) for i, box in enumerate(subBoxes) if numberOfRoots[i] != 0])

		for i, box in enumerate(subBoxes):
			box._numberOfRoots = numberOfRoots[i]

	def remove_relations(C):
		# remove itself
		try:
			boxes.remove((C, C._numberOfRoots))
		except ValueError:
			pass

		# get all direct relations
		# siblings:
		relations = C._parentBox._childBoxes 
		relations.remove(C)

		# children:
		if hasattr(C, '_childBoxes'):
			relations.extend(C._childBoxes)

		# interate over all relations
		for relation in relations:
			remove_relations(relation)

	def addRoot(root, multiplicity=None, useGuessRootSymmetry=None):
		# check that the root we have found is distinct from the ones we already have
		if not roots or np.all(abs(np.array(roots) - root) > newtonStepTol):
			# add the root to the list if it is within the original box
			if originalContour.contains(root):
				roots.append(root)
				if multiplicity is None:
					# XXX: Not the best way to determine multiplicity if this root is clustered close to others
					from .Contours import Circle
					C = Circle(root, 1e-3)
					multiplicity, = C.approximate_roots(f, df, absTol, relTol, NintAbsTol, integerTol, errStop, divMin, divMax, m, newtonStepTol, intMethod, verbose)[1]

				multiplicities.append(multiplicity.real)

			# check to see if there are any other roots implied by the given symmetry
			if guessRootSymmetry is not None:
				for x0 in guessRootSymmetry(root):
					root = iterateToRoot(x0, f, df, newtonStepTol, rootErrTol, newtonMaxIter, attemptIterBest)
					if root is not None:
						addRoot(root)

	# Add given roots 
	for guess in guessRoots:
		if hasattr(guess, '__iter__'):
			root, multiplicity = guess
			if not multiplicity_correct(f, df, root, multiplicity):
				continue

		else:
			root, multiplicity = guess, None

		# XXX: refine root with Newton-Raphson?
		
		if abs(f(root)) < rootErrTol:
			addRoot(root, multiplicity)

	# yield so that the animation shows the first frame
	totFoundRoots = sum(int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities))
	yield roots, multiplicities, boxes, totNumberOfRoots - totFoundRoots

	# print('Tot number of Roots', totNumberOfRoots)

	while boxes:
		box, numberOfRoots = boxes.pop()

		if verbose:
			print(numberOfRoots, 'roots in', box)

		# if a known root is too near this box then reverse the subdivision that created it 
		if np.any([box.distance(root) < newtonStepTol for root in roots]):
			# remove the box and any relations
			remove_relations(box)

			# put the parent box back into the list of boxes to subdivide again
			parent = box._parentBox
			boxes.append((parent, parent._numberOfRoots))

			# compute the root multiplicity directly
			approxRootMultiplicity = None

			# do not use this box again
			failedBoxes.append(box)

			continue

		# print(numberOfRoots, box)

		# if box is smaller than the newtonStepTol then just assume that the root is
		# at the center of the box, print a warning and move on
		if box.area < newtonStepTol:
			smallBoxWarning = True
			if smallBoxWarning is False:
				warnings.warn('The area of the interior of a contour containing %i is smaller than newtonStepTol!  \
					\nThe center of the box has been recored as a root of multiplicity %i but this could not be verified.  \
					\nThe same assumption will be made for future contours this small without an additional warning.  \
					\nrootErrTol may be too small.'%(numberOfRoots, numberOfRoots))
			root = box.centralPoint
			addRoot(root,  multiplicity=numberOfRoots)
			continue

		# if all the roots within the box have been located then coninue to the next box
		numberOfKnownRootsInBox = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if box.contains(root)])
		# print('box', box, 'N', numberOfRoots, 'knownN', numberOfKnownRootsInBox)
		
		if numberOfRoots == numberOfKnownRootsInBox:
			continue

		# if there are too many roots within the contour then subdivide
		# if there are any known roots within the contour then subdivide (so as not to waste resources re-approximating them)
		if numberOfKnownRootsInBox > 0 or numberOfRoots > M:
			subdivide(box)

		else:
			# approximate the roots in this box
			try:
				approxRoots, approxRootMultiplicities = box.approximate_roots(f, df, absTol, relTol, NintAbsTol, integerTol, errStop, divMin, divMax, m, newtonStepTol, intMethod, verbose)
			except MultiplicityError:
				subdivide(box)
				continue

			for approxRoot, approxRootMultiplicity in list(zip(approxRoots, approxRootMultiplicities)):
				# XXX: if the approximate root is not in this box then desregard and redo the subdivision?

				# attempt to refine the root
				root = iterateToRoot(approxRoot, f, df, newtonStepTol, rootErrTol, newtonMaxIter, attemptIterBest)

				# print('approx', approxRoot, 'refined root', root)

				if abs(f(approxRoot)) < rootErrTol and (root is None or abs(f(approxRoot)) < abs(f(root))):
					# stick with the original approximation
					root = approxRoot

				if root is not None:
					# if the root is very close to the contour then disregard this contour and compute the root multiplicity directly
					# XXX: implement a distance function returning the shortest distance from a point to any point on a contour
					if box.distance(root) < newtonStepTol:
						# remove the box and any relations
						remove_relations(box)

						# put the parent box back into the list of boxes to subdivide again
						parent = box._parentBox
						boxes.append((parent, parent._numberOfRoots))

						# compute the root multiplicity directly
						approxRootMultiplicity = None

						# do not use this box again
						failedBoxes.append(box)
					
					# if we found a root add it to the list of known roots
					addRoot(root, approxRootMultiplicity)

			# if we haven't found all the roots then subdivide further
			numberOfKnownRootsInBox = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if box.contains(root)])
			if numberOfRoots != numberOfKnownRootsInBox and box not in failedBoxes:
				subdivide(box)

		totFoundRoots = sum(int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities))
		yield roots, multiplicities, boxes, totNumberOfRoots - totFoundRoots

	# delete cache for original contour incase this contour is being reused
	for segment in originalContour.segments:
		segment._integralCache = {}
		segment._contArgCache = {}
		segment._trapValuesCache = {}

	# yield one more time so that the animation shows the final frame
	yield roots, multiplicities, boxes, totNumberOfRoots - totFoundRoots

	if totNumberOfRoots == 0:
		yield [], [], deque(), 0

@docstrings.dedent
@doc_tab_to_space
def findRoots(originalContour, f, df=None, **kwargs):
	"""
	Find all the roots of the complex analytic function f within the given contour.

	Parameters
	----------
	%(findRootsGen.parameters)s

	Returns
	-------
	result : rootResult
		A container for the roots and their multiplicities
	"""
	rootFinder = findRootsGen(originalContour, f, df, **kwargs)
	for roots, multiplicities, boxes, numberOfRemainingRoots in rootFinder:
		pass
	return RootResult(roots, multiplicities, originalContour)

