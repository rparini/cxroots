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

def subdivide(boxDeque, parentBox, parentBox_numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax, failedBoxes):
	numberOfRoots = None
	for subBoxes in parentBox.subdivisions():
		# if a box has already been used and caused an error then skip it
		failedBoxDescriptions = list(map(str, failedBoxes))
		if str(subBoxes[0]) in failedBoxDescriptions or str(subBoxes[1]) in failedBoxDescriptions:
			continue

		# XXX: if a root is near to a box then subdivide again?

		try:
			numberOfRoots = [box.count_roots(f, df, integerTol, integrandUpperBound, divMax) for box in np.array(subBoxes)]
			if parentBox_numberOfRoots == sum(numberOfRoots):
				break
		except RootError:
			# If the number of zeros within either of the new contours is not an integer then it is
			# likely that the introduced line which subdivides 'parentBox' lies on a zero.
			# To avoid this we will try to place the subdividing line at a different point along 
			# the division axis
			continue

	if numberOfRoots is None or parentBox_numberOfRoots != sum(numberOfRoots):
		# The list of subdivisions has been exhaused and still the number of enclosed zeros does not add up 
		raise RuntimeError("""Unable to subdivide box:
			\t%s
			Consider increasing the integrandUpperBound to allow contours closer to roots to be integrated.""" % parentBox)

	boxDeque.extend([(box, numberOfRoots[i]) for i, box in enumerate(subBoxes) if numberOfRoots[i] != 0])
		
def addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter, multiplicity=None):
	# check that the root we have found is distinct from the ones we already have
	if not roots or np.all(abs(np.array(roots) - root) > newtonStepTol):
		# add the root to the list if it is within the original box
		if originalContour.contains(root):
			roots.append(root)
			if multiplicity is None:
				from .Contours import Circle
				C = Circle(root, 1e-2)
				# multiplicity, = C.approximate_roots(f, df, absTol, relTol, integerTol, integrandUpperBound, divMax, rootTol=newtonStepTol)[1]
				multiplicity, = C.approximate_roots(f, df, rootTol=newtonStepTol)[1]

			multiplicities.append(multiplicity)

		# check to see if there are any other roots implied by the given symmetry
		if guessRootSymmetry is not None:
			for x0 in guessRootSymmetry(root):
				root = iterateToRoot(x0, f, df, newtonStepTol, rootErrTol, newtonMaxIter)
				if root is not None:
					addRoot(root, roots, multiplicities, originalContour, f, df, None, newtonStepTol, rootErrTol, newtonMaxIter)

def findRootsGen(originalContour, f, df=None, guessRoot=[], guessRootSymmetry=None, 
	newtonStepTol=1e-8, newtonMaxIter=20, rootErrTol=1e-12,
	absTol=1e-12, relTol=1e-12, divMax=10, integerTol=0.25, integrandUpperBound=1e3,
	M=5):
	"""
	A generator which at each step takes a contour and either finds 
	all the zeros of f within it or subdivides it further.  Based
	on the algorithm in [KB]

	Parameters
	----------
	originalContour : subclass of Contour
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
	guessRoot : list, optional
		A list of (root, multiplicity) tuples giving the already 
		known roots of the function f. 
	guessRootSymmetry : function, optional
		A function of a single complex variable, z, which 
		returns a list of all points which are expected to be 
		roots of f, given that z is a root of f.  It is assumed 
		that the symmetric roots have the same multiplicity of 
		the original root.
	newtonStepTol : float, optional
		The required accuracy of the root.
		The iterative method used to give a final value for each
		root will exit if the step size, dx, between sucessive 
		iterations satisfied abs(dx) < newtonStepTol
	newtonMaxIter : int, optional
		The iterative method used to give a final value for each
		root will exit if the number of iterations exceeds newtonMaxIter.
	rootErrTol : float, optional
		The iterative method used to give a final value for each
		root will exit if, for a point approimating the root, x, 
		abs(f(x)) < rootErrTol.
	absTol : float, optional
		The integration along a path will return a result if the
		difference between the last two iterations is less than abstol.
	relTol : float, optional
		The integration along a path will return a result if the 
		difference between the last two iterations are less than 
		relTol multiplied by the last iteration.
	divMax : int, optional
		The maximum number of divisions before the Romberg integration
		routine of a path exits.
	integerTol : float, optional
		If the result of a contour integration is expected to be 
		an integer (ie. when computing the number of zeros within 
		a contour) then the result of the last iteration of the 
		intergration must be within integerTol of an integer.
		The parameters absTol and relTol are not used in this case.
	integrandUpperBound : float, optional
		The maximum allowed value of abs(df(z)/f(z)) along a contour.  
		If abs(df(z)/f(z)) is found to exceed this value then there 
		is likely to be a root close to the contour and the
		contour is discarded and a new subdivision created.
		If integrandUpperBound is too large then integrals may take 
		a very long time to converge and it is generally be more 
		efficient to allow the rootfinding procedure to instead 
		choose another contour then spend time evaluting the 
		integral along a contour very close to a root.
	M : int, optional
		If the number of roots (including multiplicites) within a
		contour is greater than M then the contour is subdivided
		further.  M must be greater than or equal to the largest 
		multiplcity of any root.  

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
		totNumberOfRoots = originalContour.count_roots(f, df, integerTol, integrandUpperBound, divMax)
	except RuntimeError:
		raise RuntimeError("""
			Integration along the initial contour has failed.  There is likely a root on or close to the initial contour
			Try either changing the initial contour, if possible, or increasing the integrandUpperBound to allow for 
			a longer integration time.""")

	smallBoxWarning = False
	roots = []
	multiplicities = []
	failedBoxes = []

	# Add given roots 
	# XXX: check these roots and multiplcities
	for root, multiplicity in guessRoot:
		addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter)

	# print('Tot number of Roots', totNumberOfRoots)

	boxes = deque()
	boxes.append((originalContour,totNumberOfRoots))
	while boxes:
		box, numberOfRoots = boxes.pop()

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
			addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter, multiplicity=numberOfRoots)
			continue

		# if all the roots within the box have been located then coninue to the next box
		numberOfKnownRootsInBox = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if box.contains(root)])
		# print('box', box, 'N', numberOfRoots, 'knownN', numberOfKnownRootsInBox)
		# print(roots)
		# print(multiplicities)

		if numberOfRoots == numberOfKnownRootsInBox:
			continue

		# if there are too many roots within the contour then subdivide
		# if there are any known roots within the contour then subdivide (so as not to waste resources re-approximating them)
		if numberOfKnownRootsInBox > 0 or numberOfRoots > M:
			subdivide(boxes, box, numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax, failedBoxes)

		else:
			# approximate the roots in this box
			approxRoots, approxRootMultiplicities = box.approximate_roots(f, df, absTol, relTol, integerTol, integrandUpperBound, divMax, rootTol=newtonStepTol)
			# print('approxRoots', approxRoots)
			# print('approxRootsAbs', np.abs(f(np.array(approxRoots))))
			# print('approxRootMultiplicities', approxRootMultiplicities)
			for approxRoot, approxRootMultiplicity in list(zip(approxRoots, approxRootMultiplicities)):
				# XXX: if the approximate root is not in this box then desregard and redo the subdivision?

				if abs(f(approxRoot)) < rootErrTol:
					# the approximate root is good enough
					root = approxRoot
				else:
					# attempt to refine the root
					root = iterateToRoot(approxRoot, f, df, newtonStepTol, rootErrTol, newtonMaxIter)
					# print('refined root', root)

				# if the root is very close to the contour then disregard this contour and compute the root multiplicity directly
				# XXX: implement a distance function returning the shortest distance from a point to any point on a contour
				t = np.linspace(0,1,10001) 
				if np.any(np.abs(box(t) - root) < 1/integrandUpperBound):
					parent = box._parentBox
					siblings = parent._childBoxes
					siblings.remove(box)

					# remove siblings from boxes
					parent_numberOfRoots = numberOfRoots
					sibling_subBoxes = [(C, C_NumberOfRoots) for C, C_NumberOfRoots in boxes if C in siblings]
					for C, C_NumberOfRoots in sibling_subBoxes:
						parent_numberOfRoots += C_NumberOfRoots
						boxes.remove((C, C_NumberOfRoots))

					# put the parent box back into the list of boxes to subdivide again
					boxes.append((parent, parent_numberOfRoots))

					# compute the root multiplicity directly
					approxRootMultiplicity = None

					# do not use this box again
					failedBoxes.append(box)

				if root is not None:
					# if we found a root add it to the list of known roots
					addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter, approxRootMultiplicity)

			# if we haven't found all the roots then subdivide further
			numberOfKnownRootsInBox = sum([int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities) if box.contains(root)])
			if numberOfRoots != numberOfKnownRootsInBox and box not in failedBoxes:
				subdivide(boxes, box, numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax, failedBoxes)

		totFoundRoots = sum(int(round(multiplicity.real)) for root, multiplicity in zip(roots, multiplicities))
		yield roots, multiplicities, boxes, totNumberOfRoots - totFoundRoots

	if totNumberOfRoots == 0:
		yield [], [], deque(), 0

def findRoots(originalContour, f, df=None, **kwargs):
	"""
	Return a list of all roots of a given function f within a given originalContour.  
	Shares key word arguments with :func:`cxroots.RootFinder.findRootsGen`.
	"""
	rootFinder = findRootsGen(originalContour, f, df, **kwargs)
	for roots, multiplicities, boxes, numberOfRemainingRoots in rootFinder:
		pass
	return roots, multiplicities

def demo_findRoots(originalContour, f, df=None, automaticAnimation=False, returnAnim=False, **kwargs):
	"""
	An interactive demonstration of the processess used to find all the roots
	of a given function f within a given originalContour.
	Shares key word arguments with :func:`cxroots.RootFinder.findRootsGen`. 

	If automaticAnimation is False (default) then press the SPACE key 
	to step the animation forward.

	If automaticAnimation is True then the animation will play automatically
	until all the roots have been found.

	If returnAnim is true the animating object returned by matplotlib's animation.FuncAnimation
	will be returned, rather than the animation be shown.
	"""
	import matplotlib.pyplot as plt
	from matplotlib import animation
	fig = plt.gcf()
	ax = plt.gca()

	rootFinder = findRootsGen(originalContour, f, df, **kwargs)
	originalContour.plot(linecolor='k', linestyle='--')

	### XXX: show total number of roots to be found at the start
	# ax.text(0.02, 0.95, 'Zeros remaining: %i'%originalContour.count_roots(f,df,**kwargs), transform=ax.transAxes)

	def update_frame(args):
		roots, multiplicities, boxes, numberOfRemainingRoots = args
		# print(args)

		plt.cla() # clear axis
		originalContour.plot(linecolor='k', linestyle='--')
		originalContour.sizePlot()
		for box, numberOfEnclosedRoots in boxes:
			if not hasattr(box, '_color'):
				cmap = plt.get_cmap('jet')
				box._color = cmap(np.random.random())
			
			plt.text(box.centralPoint.real, box.centralPoint.imag, numberOfEnclosedRoots)
			box.plot(linecolor=box._color)

		plt.scatter(np.real(roots), np.imag(roots))

		rootsLabel = ax.text(0.02, 0.95, 'Zeros remaining: %i'%numberOfRemainingRoots, transform=ax.transAxes)

		fig.canvas.draw()


	if returnAnim:
		return animation.FuncAnimation(fig, update_frame, frames=list(rootFinder), interval=500, repeat_delay=2000)

	elif automaticAnimation:
		ani = animation.FuncAnimation(fig, update_frame, frames=rootFinder, interval=500)

	else:
		def draw_next(event):
			if event.key == ' ':
				update_frame(next(rootFinder))

		fig.canvas.mpl_connect('key_press_event', draw_next)

	plt.show()

	return roots, multiplicities

def showRoots(originalContour, f, df=None, **kwargs):
	"""
	Plots all roots of a given function f within a given originalContour.  
	Shares key word arguments with :func:`cxroots.RootFinder.findRootsGen`.
	"""
	import matplotlib.pyplot as plt
	originalContour.plot(linecolor='k', linestyle='--')
	roots = findRoots(originalContour, f, df, **kwargs)
	plt.scatter(np.real(roots), np.imag(roots), color='k', marker='x')
	plt.show()

