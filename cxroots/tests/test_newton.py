import pytest

from cxroots.IterativeMethods import newton
from numpy import pi, cos, sin

def test_newton():
	# result from keisan online calculator: http://keisan.casio.com/exec/system/1244946907
	f  = lambda x: cos(x)-x
	df = lambda x: -sin(x)-1

	iterations = []
	callback = lambda x, dx, y, iteration: iterations.append(x)
	x, err = newton(pi/4, f, df, callback=callback)
	iterations.append(x)
	
	correct_iterations = [0.73953613351523830094,
						  0.7390851781060101829533,
			  			  0.7390851332151610866198,
			  			  0.7390851332151606416553]

	assert iterations == pytest.approx(correct_iterations)

