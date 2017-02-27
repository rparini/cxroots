from __future__ import division
# import scipy.optimize

def iterateToRoot(x0, f, df=None, steptol=1e-8, roottol=1e-14, maxIter=20):
	# iterate to a root using initial point x0
	if df is not None:
		try:
			# uses Newton-Raphson method if f and df are given.

			# SciPy implementation
			# root = scipy.optimize.newton(f, x0, df, tol=steptol, maxiter=maxIter)
			# err = abs(f(root))
			
			root, err = newton(x0, f, df, steptol, maxIter)

		except (RuntimeError, OverflowError):
			return None
	else:
		# if only f is given then use secant method
		# XXX: Perhaps implement Muller's method to use in the case were df is unavailable? 
		x1, x2 = x0, x0*(1 + 1e-4) + 1e-4
		root, err = secant(x1, x2, f, steptol=steptol, maxIter=maxIter)

	if err < roottol:
		return root

def newton(x0, f, df, steptol=1e-8, maxIter=20, callback=None):
	"""
	Find an approximation to a point xf such that f(xf)=0 for a 
	scalar function f using Newton–Raphson iteration starting at 
	the point x0.

	Parameters
	----------
	x0 : float or complex
		Initial point for Newton iteration, should be as close as
		possible to a root of f
	f : function
		Function of a single variable f(x)
	df : function
		Function of a single variable, df(x), providing the
		derivative of the function f(x) at the point x
	steptol: float, optional
		Routine will end if the step size, dx, between sucessive
		iterations of x satisfies abs(dx) < steptol
	maxIter : int, optional
		Routine ends after maxIter iterations
	callback : function, optional
		After each iteration the supplied function 
		callback(x, dx, f(x), iteration) will be called where 'x' is the current iteration 
		of the estimated root, 'dx' is the step size between the previous 
		and current 'x' and 'iteration' the number of iterations that have been taken.  
		If the callback function evaluates to True then the routine will end

	Returns
	-------
	xf : float
		The approximation to a root of f
	rooterr : float
		The error of the original function at xf, abs(f(xf))
	"""

	# XXX: Could use deflated polynomials to ensure that known roots are not found again?
	
	x, y = x0, f(x0)
	for iteration in range(maxIter):
		dx = -y/df(x)
		x += dx
		y  = f(x)

		if callback is not None and callback(x, dx, y, iteration+1):
			break

		if abs(dx) < steptol:
			break

	return x, abs(y)

def secant(x1, x2, f, steptol=1e-10, maxIter=30, callback=None):
	"""
	Find an approximation to a point xf such that f(xf)=0 for a 
	scalar function f using the secant method.  The method requires
	two initial points x1 and x2, ideally close to a root
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
		Function of a single variable f(x)
	steptol: float, optional
		Routine will end if the step size, dx, between sucessive
		iterations of x satisfies abs(dx) < steptol
	maxIter : int, optional
		Routine ends after maxIter iterations
	callback : function, optional
		After each iteration the supplied function 
		callback(x, dx, f(x), iteration) will be called where 'x' is the current iteration 
		of the estimated root, 'dx' is the step size between the previous 
		and current 'x' and 'iteration' the number of iterations that have been taken.  
		If the callback function evaluates to True then the routine will end

	Returns
	-------
	xf : float
		The approximation to a root of f
	rooterr : float
		The error of the original function at xf, abs(f(xf))
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

		if abs(dx) < steptol:
			break

	return x2, abs(y2)
