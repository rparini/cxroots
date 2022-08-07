import numpy as np
import pytest

from cxroots import Circle, Rectangle
from cxroots.tests.approx_equal import roots_approx_equal

funcs = [
    pytest.param(
        [3, 3.01, 3.02, 8, 8.02, 8 + 0.01j], [1, 1, 1, 1, 1, 1], id="cluster_10^-2"
    ),
    pytest.param(
        [3, 3.001, 3.002, 8, 8.002, 8 + 0.001j], [1, 1, 1, 1, 1, 1], id="cluster_10^-3"
    ),
    pytest.param(
        [3, 3.0001, 3.0002, 8, 8.0002, 8 + 0.0001j],
        [1, 1, 1, 1, 1, 1],
        id="cluster_10^-4",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        [3, 3.00001, 3.00002, 8, 8.00002, 8 + 0.00001j],
        [1, 1, 1, 1, 1, 1],
        id="cluster_10^-5",
        marks=[
            pytest.mark.slow,
            pytest.mark.xfail(reason="Cluster of roots too tight"),
        ],
    ),
]

contours = [
    pytest.param(Rectangle([2, 9], [-1, 1]), id="rect"),
    pytest.param(Circle(0, 8.5), id="circle"),
]


@pytest.mark.parametrize("contour", contours)
@pytest.mark.parametrize("roots,multiplicities", funcs)
def test_rootfinding_df(contour, roots, multiplicities):
    def f(z):
        return np.prod([z - r for r in roots], axis=0)

    def df(z):
        return np.sum(
            [
                np.prod([z - r for r in np.delete(roots, i)], axis=0)
                for i in range(len(roots))
            ],
            axis=0,
        )

    roots_approx_equal(contour.roots(f, df), (roots, multiplicities))


@pytest.mark.slow
@pytest.mark.parametrize("contour", contours)
@pytest.mark.parametrize("roots,multiplicities", funcs)
def test_rootfinding_f(contour, roots, multiplicities):
    def f(z):
        return np.prod([z - r for r in roots], axis=0)

    roots_approx_equal(contour.roots(f), (roots, multiplicities))
