from __future__ import division
from numpy import inf

def iterateToRoot(x0, f, df=None, steptol=1e-12, roottol=1e-12, maxIter=20, 
	attemptBest=False, verbose=False, callback=None):
	"""
	Starting with initial point x0 iterate to a root of f. This function 
	is called during the rootfinding process to refine any roots found.
	If df is given then the Newton-Raphson method, :func:`~cxroots.IterativeMethods.newton`,
	will be used, otherwise Muller's method, :func:`~cxroots.IterativeMethods.muller`, will be used instead.

	Parameters
	----------
	x0 : complex
		An initial point for the iteration.
	f : function
		Function of a single variable which we seek to find a root of.
	df : function, optional
		The derivative of f.
	steptol: float, optional
		The routine ends if the step size, dx, between sucessive
		iterations satisfies abs(dx) < steptol and attemptBest is False.
	roottol: float, optional
		The routine ends if abs(f(x)) < roottol and attemptBest is False.
	maxIter : int, optional
		The routine ends after maxIter iterations.
	attemptBest : bool, optional
		If True then routine ends if the error of the previous iteration, 
		x0, was at least as good as the current iteration, x, in the 
		sense that abs(f(x)) >= abs(f(x0)) and the previous iteration 
		satisfied either abs(dx0) < steptol or abs(f(x0)) < roottol.  In 
		this case the previous iteration is returned as the approximation 
		of the root.
	verbose : bool, optional
		Print x, dx and f(x) at each step of the iteration.
	callback : function, optional
		After each iteration callback(x, dx, f(x), iteration) will be 
		called where 'x' is the current iteration of the estimated root, 
		'dx' is the step size between the previous and current 'x' and 
		'iteration' the number of iterations that have been taken.  If 
		the callback function evaluates to True then the routine will end.

	Returns
	-------
	complex
		An approximation for a root of f.  If the rootfinding was 
		unsucessful then None will be returned instead.
	"""
	if verbose: print('Refining root:', x0)

	if df is not None:
		try:
			root, err = newton(x0, f, df, steptol, 0, maxIter, attemptBest, verbose, callback)
		except (RuntimeError, OverflowError):
			return None
	else:
		# Muller's method:
		f_muller = lambda z: complex(f(z))
		x1, x2, x3 = x0, x0*(1 + 1e-8) + 1e-8, x0*(1 - 1e-8) - 1e-8
		root, err = muller(x1, x2, x3, f_muller, steptol, 0, maxIter, attemptBest, verbose, callback)

	if err < roottol:
		return root

def muller(x1, x2, x3, f, steptol=1e-12, roottol=1e-12, maxIter=20, 
	attemptBest=False, verbose=False, callback=None):
	"""
	A wrapper for mpmath's implementation of Muller's method.  

	Parameters
	----------
	x1 : float or complex
		An initial point for iteration, should be close to a root of f.
	x2 : float or complex
		An initial point for iteration, should be close to a root of f.  
		Should not equal x1.
	x3 : float or complex
		An initial point for iteration, should be close to a root of f.  
		Should not equal x1 or x2.
	f : function
		Function of a single variable which we seek to find a root of.
	steptol: float, optional
		The routine ends if the step size, dx, between sucessive
		iterations satisfies abs(dx) < steptol and attemptBest is False.
	roottol: float, optional
		The routine ends if abs(f(x)) < roottol and attemptBest is False.
	maxIter : int, optional
		The routine ends after maxIter iterations.
	attemptBest : bool, optional
		If True then routine ends if the error of the previous iteration, 
		x0, was at least as good as the current iteration, x, in the 
		sense that abs(f(x)) >= abs(f(x0)) and the previous iteration 
		satisfied either abs(dx0) < steptol or abs(f(x0)) < roottol.  In 
		this case the previous iteration is returned as the approximation 
		of the root.
	verbose : bool, optional
		Print x, dx and f(x) at each step of the iteration.
	callback : function, optional
		After each iteration callback(x, dx, f(x), iteration) will be 
		called where 'x' is the current iteration of the estimated root, 
		'dx' is the step size between the previous and current 'x' and 
		'iteration' the number of iterations that have been taken.  If 
		the callback function evaluates to True then the routine will end.

	Returns
	-------
	complex
		The approximation to a root of f.
	float
		abs(f(x)) where x is the final approximation for the root of f.
	"""
	from mpmath import mp, mpmathify
	from mpmath.calculus.optimization import Muller

	# mpmath insists on functions accepting mpc
	f_mpmath = lambda z: mpmathify(f(complex(z)))

	mull = Muller(mp, f_mpmath, (x1, x2, x3), verbose=False)
	iteration = 0
	x0 = x3

	x, err = x0, abs(f(x0))
	err0, dx0 = inf, inf
	try:
		for x, dx in mull:
			err = abs(f_mpmath(x))

			if verbose: print(iteration, 'x', x, '|f(x)|', err, 'dx', dx)

			if callback is not None and callback(x, dx, err, iteration+1):
				break

			if not attemptBest and (abs(dx) < steptol or err < roottol) or iteration > maxIter:
				break

			if attemptBest and (abs(dx0) < steptol or err0 < roottol) and err >= err0:
				# The previous iteration was a better appproximation the current one so  
				# assume that that was as close to the root as we are going to get.
				x, err = x0, err0
				break

			iteration += 1
			x0 = x

			if attemptBest:
				# record previous error for comparison
				dx0, err0 = dx, err

	except ZeroDivisionError:
		# ZeroDivisionError comes up if the error is evaluated to be zero
		pass

	if verbose: print('Final approximation: x=', complex(x), '|f(x)|=', float(err))

	# cast mpc and mpf back to regular complex and float
	return complex(x), float(err)


def newton(x0, f, df, steptol=1e-12, roottol=1e-12, maxIter=20, 
	attemptBest=False, verbose=False, callback=None):
	"""
	Find an approximation to a point xf such that f(xf)=0 for a 
	scalar function f using Newton-Raphson iteration starting at 
	the point x0.

	Parameters
	----------
	x0 : float or complex
		Initial point for Newton iteration, should be as close as
		possible to a root of f
	f : function
		Function of a single variable which we seek to find a root of.
	df : function
		Function of a single variable, df(x), providing the
		derivative of the function f(x) at the point x
	steptol: float, optional
		The routine ends if the step size, dx, between sucessive
		iterations satisfies abs(dx) < steptol and attemptBest is False.
	roottol: float, optional
		The routine ends if abs(f(x)) < roottol and attemptBest is False.
	maxIter : int, optional
		The routine ends after maxIter iterations.
	attemptBest : bool, optional
		If True then routine ends if the error of the previous iteration, 
		x0, was at least as good as the current iteration, x, in the 
		sense that abs(f(x)) >= abs(f(x0)) and the previous iteration 
		satisfied either abs(dx0) < steptol or abs(f(x0)) < roottol.  In 
		this case the previous iteration is returned as the approximation 
		of the root.
	verbose : bool, optional
		Print x, dx and f(x) at each step of the iteration.
	callback : function, optional
		After each iteration callback(x, dx, f(x), iteration) will be 
		called where 'x' is the current iteration of the estimated root, 
		'dx' is the step size between the previous and current 'x' and 
		'iteration' the number of iterations that have been taken.  If 
		the callback function evaluates to True then the routine will end.

	Returns
	-------
	complex
		The approximation to a root of f.
	float
		abs(f(x)) where x is the final approximation for the root of f.
	"""	
	x, y = x0, f(x0)
	dx0, y0 = inf, y
	for iteration in range(maxIter):
		dx = -y/df(x)
		x += dx
		y  = f(x)

		if verbose: print('x', x, 'f(x)', y, 'dx', dx)

		if callback is not None and callback(x, dx, y, iteration+1):
			break

		if not attemptBest and (abs(dx) < steptol or abs(y) < roottol):
			break

		if attemptBest and (abs(dx0) < steptol or abs(y0) < roottol) and abs(y) > abs(y0):
			break

		if attemptBest:
			# store previous dx and y
			dx0, y0 = dx, y

	if verbose: print('Final approximation: x=', x, '|f(x)|=', abs(y))

	return x, abs(y)

def secant(x1, x2, f, steptol=1e-12, roottol=1e-12, maxIter=30, callback=None):
	"""
	Find an approximation to a point xf such that f(xf)=0 for a 
	scalar function f using the secant method.  The method requires
	two initial points x1 and x2, ideally close to a root,
	and proceeds iteratively.

	Parameters
	----------
	x1 : float or complex
		An initial point for iteration, should be close to a 
		root of f.
	x2 : float or complex
		An initial point for iteration, should be close to a 
		root of f.  Should not equal x1.
	f : function
		Function of a single variable which we seek to find a root of.
	steptol: float, optional
		The routine ends if the step size, dx, between sucessive
		iterations satisfies abs(dx) < steptol and attemptBest is False.
	roottol: float, optional
		The routine ends if abs(f(x)) < roottol and attemptBest is False.
	maxIter : int, optional
		The routine ends after maxIter iterations.
	callback : function, optional
		After each iteration callback(x, dx, f(x), iteration) will be 
		called where 'x' is the current iteration of the estimated root, 
		'dx' is the step size between the previous and current 'x' and 
		'iteration' the number of iterations that have been taken.  If 
		the callback function evaluates to True then the routine will end.

	Returns
	-------
	complex
		The approximation to a root of f.
	float
		abs(f(x)) where x is the final approximation for the root of f.
	"""
	# As in "Numerical Recipies 3rd Edition" pick the bound with the 
	# smallest function value as the most recent guess
	y1, y2 = f(x1), f(x2)
	if abs(y1) < abs(y2):
		x1, x2 = x2, x1
		y1, y2 = y2, y1

	for iteration in range(maxIter):
		dx =  -(x2-x1)*y2/(y2-y1)
		x1, x2 = x2, x2 + dx
		y1, y2 = y2, f(x2)

		if callback is not None and callback(x2, dx, y2, iteration+1):
			break

		if abs(dx) < steptol or abs(y2) < roottol:
			break

	return x2, abs(y2)
