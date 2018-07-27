import pytest
from numpy import exp, sin, cos

from cxroots import Circle
from cxroots.tests.ApproxEqual import roots_approx_equal

@pytest.mark.parametrize('symmetry', [
	pytest.param(lambda z: [z.conjugate()], id='right_symmetry'),
	pytest.param(lambda z: [z+1], 		   	id='wrong_symmetry')
])
def test_guess_symmetry(symmetry):
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

