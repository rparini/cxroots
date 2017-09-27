import scipy

def find_multiplicity(root, f, df=None, rootErrTol=1e-8, dx=1e-8):
	"""
	given a root of the function f determine the multiplicity
	of the root numerically by evaluating the derivatives of f
	"""
	i, f_root = 0, f(root)
	while abs(f_root) < rootErrTol:
		i += 1
		if df == None:
			f_root = scipy.misc.derivative(f, root, dx=dx, n=i, order=2*i+1)
		else:
			if i == 1:
				f_root = df(root)
			else:
				f_root = scipy.misc.derivative(df, root, dx=dx, n=i-1, order=2*(i-1)+1)
	return i
