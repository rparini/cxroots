import pytest
from numpy import exp, sin, cos
import numpy as np

from cxroots import Circle
from cxroots.tests.ApproxEqual import roots_approx_equal

@pytest.mark.parametrize('symmetry', [
	pytest.param(lambda z: [z.conjugate()], id='right_symmetry'),
	pytest.param(lambda z: [z+1], 		   	id='wrong_symmetry')
])
def test_guess_symmetry_1(symmetry):
	C = Circle(0, 3)
	f = lambda z: z**4 + z**3 + z**2 + z

	roots = [0,-1,1j,-1j]
	multiplicities = [1,1,1,1]

	roots_approx_equal(C.roots(f, verbose=True, guessRootSymmetry=symmetry), (roots, multiplicities))


@pytest.mark.parametrize('guesses', [
	pytest.param([2.5],    id='right_root_guess'),
	pytest.param([2.5, 3], id='wrong_root_guess')
])
def test_guess_root(guesses):
	C = Circle(0, 3)
	f = lambda z: (z-2.5)**2 * (exp(-z)*sin(z/2.) - 1.2*cos(z))

	roots = [2.5,
			 1.44025113016670301345110737, 
			 -0.974651035111059787741822566 - 1.381047768247156339633038236j,
			 -0.974651035111059787741822566 + 1.381047768247156339633038236j]
	multiplicities = [2,1,1,1]

	roots_approx_equal(C.roots(f, guessRoots=[2.5], verbose=True), (roots, multiplicities))

@pytest.mark.parametrize('usedf', [
	pytest.param(True,  id='with_df'),
	pytest.param(False, id='wihout_df')
])
def test_guess_symmetry_2(usedf):
	C = Circle(0, 1.5)
	f = lambda z: z**27-2*z**11+0.5*z**6-1
	df = (lambda z: 27*z**26-22*z**10+3*z**5) if usedf else None

	symmetry = lambda z: [z.conjugate()]

	roots = [-1.03509521179240, 
		  	 -0.920332541459108, 
		  	 1.05026721944263, 
		  	 -0.983563736801535 - 0.382365167035741j, 
		  	 -0.983563736801535 + 0.382365167035741j, 
		  	 -0.792214346729517 - 0.520708613101932j, 
		  	 -0.792214346729517 + 0.520708613101932j, 
		  	 -0.732229626596468 - 0.757345327222341j, 
		  	 -0.732229626596468 + 0.757345327222341j, 
		  	 -0.40289002582335 - 0.825650446354661j, 
		  	 -0.40289002582335 + 0.825650446354661j, 
		  	 -0.383382611408318 - 0.967939747947639j, 
		  	 -0.383382611408318 + 0.967939747947639j, 
		  	 -0.02594227096144 - 1.05524415820652j, 
		  	 -0.02594227096144 + 1.05524415820652j, 
		  	 0.160356899544475 - 0.927983420797727j, 
		  	 0.160356899544475 + 0.927983420797727j, 
		  	 0.41133738621461 - 0.967444751898913j, 
		  	 0.41133738621461 + 0.967444751898913j, 
		  	 0.576737152896681 - 0.719511178392941j, 
		  	 0.576737152896681 + 0.719511178392941j, 
		  	 0.758074415348703 - 0.724716122470435j, 
		  	 0.758074415348703 + 0.724716122470435j, 
		  	 0.903278407433416 - 0.22751872334709j, 
		  	 0.903278407433416 + 0.22751872334709j, 
		  	 0.963018623787179 - 0.427294816877434j, 
		  	 0.963018623787179 + 0.427294816877434j]
	
	multiplicities = np.ones_like(roots)

	roots_approx_equal(C.roots(f, df, verbose=True, guessRootSymmetry=symmetry), (roots, multiplicities))

