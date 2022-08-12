import pytest
from numpy import cos, pi, sin

from cxroots.iterative_methods import newton


@pytest.mark.parametrize("refine_roots_beyond_tol", [True, False])
def test_newton(refine_roots_beyond_tol):
    # result from online calculator: http://keisan.casio.com/exec/system/1244946907

    def f(x):
        return x - cos(x)

    def df(x):
        return 1 + sin(x)

    iterations = []

    def callback(x, dx, y, iteration):
        iterations.append((x, dx, y))

    x0 = pi / 4
    step_tol = 1e-10
    newton(
        x0,
        f,
        df,
        step_tol=step_tol,
        callback=callback,
        refine_roots_beyond_tol=refine_roots_beyond_tol,
    )

    correct_x = (
        0.7395361335152383009399981259049958,
        0.7390851781060101829533431920344887,
        0.7390851332151610866197574459477856,
        0.7390851332151606416553120876739171,
        0.7390851332151606416553120876738734,
        0.7390851332151606416553120876738734,
    )

    correct_iterations = [
        (x, x - x0 if i == 0 else x - correct_x[i - 1], f(x))
        for i, x in enumerate(correct_x)
    ]

    for i, iteration in enumerate(iterations):
        assert iteration == pytest.approx(correct_iterations[i])

    # dx should be less than step_tol
    assert abs(iterations[-1][1]) < step_tol

    if not refine_roots_beyond_tol:
        # we should have stopped as soon as step_tol was reached to the iteration
        # before the last should not satisfy the step_tol
        assert abs(iterations[-2][1]) > step_tol
