from __future__ import division
import numpy as np
import cmath
from scipy import pi, exp, sin
import scipy
from numpy.random import uniform
import itertools
from collections import deque
import logging

from .IterativeMethods import iterateToRoot
from .CountRoots import prod

def subdivide(boxDeque, parentBox, parentBox_numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax):
	for subBoxes in parentBox.subdivisions():
		try:
			numberOfRoots = [box.count_roots(f, df, integerTol, integrandUpperBound) for box in np.array(subBoxes)]
			if parentBox_numberOfRoots == sum(numberOfRoots):
				break
		except RuntimeError:
			# If the number of zeros within either of the new contours is not an integer then it is
			# likely that the introduced line which subdivides 'parentBox' lies on a zero.
			# To avoid this we will try to place the subdividing line at a different point along 
			# the division axis
			continue

	if parentBox_numberOfRoots != sum(numberOfRoots):
		# The list of subdivisions has been exhaused and still the number of enclosed zeros does not add up 
		raise RuntimeError("""Unable to subdivide box:
			\t%s
			Consider increasing the integrandUpperBound to allow contours closer to roots to be integrated.""" % parentBox)

	boxDeque.extend([(box, numberOfRoots[i]) for i, box in enumerate(subBoxes) if numberOfRoots[i] != 0])
		
def addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter, multiplicity=None):
	# check that the root we have found is distinct from the ones we already have
	if not roots or np.all(abs(np.array(roots) - root) > newtonStepTol):
		# XXX: compute multiplicity of the root if not given
		if multiplicity is None:
			pass

		# add the root to the list if it is within the original box
		if originalContour.contains(root):
			roots.append(root)
			multiplicities.append(multiplicity)

		# check to see if there are any other roots implied by the given symmetry
		if guessRootSymmetry is not None:
			for x0 in guessRootSymmetry(root):
				root = iterateToRoot(x0, f, df, newtonStepTol, rootErrTol, newtonMaxIter)
				if root is not None:
					addRoot(root, roots, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter)


def findRootsGen(originalContour, f, df=None, guessRoot=[], guessRootSymmetry=None, 
	newtonStepTol=1e-8, newtonMaxIter=20, rootErrTol=1e-12, iterativeTries=20,
	absTol=1e-12, relTol=1e-12, divMax=20, integerTol=0.25, integrandUpperBound=1e3,
	M=5):
	"""
	A generator which at each step takes a contour and either finds 
	all the zeros of f within it or subdivides it further.

	Parameters
	----------
	originalContour : subclass of Contour
		The contour C which bounds the region in which all the 
		roots of f(z) are sought
	f : function
		A function of a single complex variable f(z) which is 
		analytic within C and has no poles or roots on C.
		NOTE: Currently required that the function f(z) has only 
		simple roots in C
	df : function, optional
		Function of a single complex variable.
		The derivative of the function f(z).  If given then the 
		number of zeros within a contour will be computed as the 
		integral of df(z)/(2j*pi*f(z)) around the contour.
		If df is not given then the integral will instead be 
		computed using the diffrence in the argument of f(z) 
		continued around the contour.
	guessRoot : list, optional
		A list of suspected roots of the function f which lie 
		within the initial contour C.  Each element of the list 
		will be used as the initial point of an iterative 
		root-finding method so they need not be entirely 
		accurate.
	guessRootSymmetry : function, optional
		A function of a single complex variable, z, which returns 
		a list of all points which are expected to be roots of f, 
		given that z is a root of f.
	newtonStepTol : float, optional
		The iterative method used to give a final value for each
		root will exit if the step size, dx, between sucessive 
		iterations satisfied abs(dx) < newtonStepTol
	newtonMaxIter : int, optional
		The iterative method used to give a final value for each
		root will exit if the number of iterations exceeds newtonMaxIter
	rootErrTol : float, optional
		For a point, x, to be confirmed as a root abs(f(x)) < rootErrTol
	iterativeTries : int, optinal
		The number of times an iterative method with a random start point
		should be used to find the root within a contour containing a single
		root before the contour is subdivided again.
	integerTol : float, optional
		The numerical evaluation of the Cauchy integral will return a result
		if the result of the last two iterations differ by less than integerTol
		and the result of the last iteration is within integerTol of an integer.
		Since the Cauchy integral must be an integer it is only necessary to
		distinguish which integer the inegral is convering towards.  For this
		reason the integerTol can be set fairly large.
	integrandUpperBound : float, optional
		The maximum allowed value of abs(df(z)/f(z)).  If abs(df(z)/f(z)) exceeds this 
		value then a RuntimeError is raised.  If integrandUpperBound is too large then 
		integrals may take a very long time to converge and it is generally be more 
		efficient to allow the rootfinding procedure to instead choose another contour 
		then spend time evaluting the integral along a contour very close to a root.

	Yields
	------
	tuple
		All currently known roots of f(z) within the contour C
	tuple
		All the contours which still contain roots
	int
		Remaining number of roots to be found within the contour
	"""
	try:
		# total number of zeros, including multiplicities
		totNumberOfRoots = originalContour.count_roots(f, df, integerTol, integrandUpperBound)
	except RuntimeError:
		raise RuntimeError("""
			Integration along the initial contour has failed.  There is likely a root on or close to the initial contour
			Try either changing the initial contour, if possible, or increasing the integrandUpperBound to allow for 
			a longer integration time.""")

	loggedIterativeWarning = False
	roots = []
	multiplicities = []

	# check to see if the guesses we were passed are roots
	for root in guessRoot:
		addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter)

	boxes = deque()
	boxes.append((originalContour,totNumberOfRoots))
	while boxes:
		box, numberOfRoots = boxes.pop()


		if numberOfRoots > M:
			subdivide(boxes, box, numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax)

		else:
			# approximate the roots in this box
			approxRoots, approxRootMultiplicities = box.approximate_roots(f, df, absTol, relTol, integerTol, integrandUpperBound, divMax)
			for approxRoot, multiplicity in list(zip(approxRoots, approxRootMultiplicities)):
				# if all the roots within the box have been located then just exit
				if numberOfRoots == sum([root*multiplicity for root, multiplicity in zip(roots, multiplicities) if box.contains(root)]):
					break

				if abs(f(approxRoot)) < rootErrTol:
					# the approximate root is good enough
					root = approxRoot
				else:
					# attempt to refine the root
					root = iterateToRoot(approxRoot, f, df, newtonStepTol, rootErrTol, newtonMaxIter)

				if root is not None:
					# if we found a root add it to the list of known roots
					addRoot(root, roots, multiplicities, originalContour, f, df, guessRootSymmetry, newtonStepTol, rootErrTol, newtonMaxIter, multiplicity)

			# if we haven't found all the roots then subdivide further
			knownRootsInBox = [root for root in roots if box.contains(root)]
			if len(knownRootsInBox) != numberOfDistinctRoots:
				subdivide(boxes, box, numberOfRoots, f, df, absTol, relTol, integerTol, integrandUpperBound, divMax)

		yield roots, multiplicities, boxes, totNumberOfRoots - len(roots)

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
	rootFinder = findRootsGen(originalContour, f, df, **kwargs)

	originalContour.plot(linecolor='k', linestyle='--')

	fig = plt.gcf()
	ax = plt.gca()

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
			
			plt.text(box.centerPoint.real, box.centerPoint.imag, numberOfEnclosedRoots)
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

