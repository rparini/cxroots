"""
References
----------
[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable"
    M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
[DL] "A Numerical Method for Locating the Zeros of an Analytic function",
    L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
[KB] "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel,
    Springer 2000
"""

import unittest

import numpy as np
import pytest
from numpy import cos, exp, sin, sqrt

from cxroots import Annulus, Circle, Rectangle
from cxroots.derivative import find_multiplicity
from cxroots.tests.approx_equal import roots_approx_equal


class RootfindingTests(object):
    # These attributes need to be defined by test case subclasses
    C = None
    f = None
    df = None
    roots = None
    multiplicities = None

    def test_rootfinding_romb_df(self):
        roots_approx_equal(
            self.C.roots(self.f, self.df, int_method="romb"),
            (self.roots, self.multiplicities),
            decimal=10,
        )

    def test_rootfinding_romb_f(self):
        roots_approx_equal(
            self.C.roots(self.f, int_method="romb"),
            (self.roots, self.multiplicities),
            decimal=10,
        )

    def test_rootfinding_quad_df(self):
        roots_approx_equal(
            self.C.roots(self.f, self.df, int_method="quad"),
            (self.roots, self.multiplicities),
            decimal=10,
        )

    def test_rootfinding_quad_f(self):
        roots_approx_equal(
            self.C.roots(self.f, int_method="quad"),
            (self.roots, self.multiplicities),
            decimal=10,
        )


class MultiplicityTests(object):
    # These attributes need to be defined by test case subclasses
    f = None
    df = None
    roots = None
    multiplicities = None

    def test_multiplicity_f(self):
        # Check that if only the root is given then the multiplcity could be computed
        for i, root in enumerate(self.roots):
            assert find_multiplicity(root, self.f, df=None) == self.multiplicities[i]

    def test_multiplicity_df(self):
        # Check that if only the root is given then the multiplcity could be computed
        for i, root in enumerate(self.roots):
            assert find_multiplicity(root, self.f, df=self.df) == self.multiplicities[i]


class TestRootfindingNoRoots(unittest.TestCase, RootfindingTests):
    def setUp(self):
        self.C = Annulus(1, [1, 2])
        self.f = lambda z: (z - 1) ** 3
        self.df = lambda z: 3 * (z - 1) ** 2

        self.roots = []
        self.multiplicities = []


class TestRootfindingPoly1(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        self.C = Rectangle([-2, 2], [-2, 2])
        self.f = lambda z: z**3 * (z - 1.2) ** 2
        self.df = lambda z: 3 * (z) ** 2 * (z - 1.2) ** 2 + 2 * z**3 * (z - 1.2)

        self.roots = [0, 1.2]
        self.multiplicities = [3, 2]


class TestRootfindingPoly2(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        self.C = Annulus(0, [0.5, 2.5])
        self.f = lambda z: (z - 2) ** 2 * (z - 1) ** 5
        self.df = lambda z: 2 * (z - 2) * (z - 1) ** 5 + 5 * (z - 2) ** 2 * (z - 1) ** 4

        self.roots = [1, 2]
        self.multiplicities = [5, 2]


class TestRootfinding141(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.1 from [KB]
        self.C = Circle(0, 3)
        e = 1e-2
        self.f = lambda z: (z - e) * (1 + (z - sqrt(3)) ** 2)
        self.df = lambda z: (1 + (z - sqrt(3)) ** 2) + (z - e) * 2 * (z - sqrt(3))

        self.roots = [e, sqrt(3) + 1j, sqrt(3) - 1j]
        self.multiplicities = [1, 1, 1]


class TestRootfinding142(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.2 from [KB]
        self.C = Circle(0, 2)
        self.f = lambda z: exp(3 * z) + 2 * z * cos(z) - 1
        self.df = lambda z: 3 * exp(3 * z) + 2 * cos(z) - 2 * z * sin(z)

        self.roots = [
            0,
            -1.844233953262213,
            0.5308949302929305 + 1.33179187675112098j,
            0.5308949302929305 - 1.33179187675112098j,
        ]
        self.multiplicities = [1, 1, 1, 1]


class TestRootfinding142b(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.2 from [KB] with a rectangular initial contour
        self.C = Rectangle([-2, 2], [-2, 2])
        self.f = lambda z: exp(3 * z) + 2 * z * cos(z) - 1
        self.df = lambda z: 3 * exp(3 * z) + 2 * cos(z) - 2 * z * sin(z)

        self.roots = [
            0,
            -1.844233953262213,
            0.5308949302929305 + 1.33179187675112098j,
            0.5308949302929305 - 1.33179187675112098j,
        ]
        self.multiplicities = [1, 1, 1, 1]


class TestRootfinding143(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.3 from [KB]
        self.C = Circle(0, 5)
        self.f = lambda z: z**2 * (z - 1) * (z - 2) * (z - 3) * (z - 4) + z * sin(z)
        self.df = (
            lambda z: 2 * z * (3 * z**4 - 25 * z**3 + 70 * z**2 - 75 * z + 24)
            + sin(z)
            + z * cos(z)
        )

        self.roots = [
            0,
            1.18906588973011365517521756,
            1.72843498616506284043592924,
            3.01990732809571222812005354,
            4.03038191606046844562845941,
        ]
        self.multiplicities = [2, 1, 1, 1, 1]


class TestRootfinding144(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.4 from [KB]
        self.C = Circle(0, 3)
        self.f = lambda z: (z * (z - 2)) ** 2 * (
            exp(2 * z) * cos(z) + z**3 - 1 - sin(z)
        )
        self.df = (
            lambda z: 2 * z * (z - 2) ** 2 * (exp(2 * z) * cos(z) + z**3 - 1 - sin(z))
            + 2 * (z - 2) * z**2 * (exp(2 * z) * cos(z) + z**3 - 1 - sin(z))
            + (z * (z - 2)) ** 2
            * (2 * exp(2 * z) * cos(z) - exp(2 * z) * sin(z) + 3 * z**2 - cos(z))
        )

        self.roots = [
            -0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
            -0.4607141197289707542294459477 + 0.6254277693477682516688207854j,
            0,
            2,
            1.66468286974551654134568653,
        ]
        self.multiplicities = [1, 1, 3, 2, 1]


class TestRootfinding145(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.5 from [KB]
        self.C = Circle(0, 11)
        self.f = lambda z: np.prod([z - k for k in range(1, 11)], axis=0)
        self.df = lambda z: np.sum(
            [
                np.prod([z - k for k in range(1, 11) if k != m], axis=0)
                for m in range(1, 11)
            ],
            axis=0,
        )

        self.roots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.multiplicities = np.ones(10)


class TestRootfinding145b(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.4.5 from [KB] with a rectangular initial contour
        self.C = Rectangle([-1, 11], [-1, 1])
        self.f = lambda z: np.prod([z - k for k in range(1, 11)], axis=0)
        self.df = lambda z: np.sum(
            [
                np.prod([z - k for k in range(1, 11) if k != m], axis=0)
                for m in range(1, 11)
            ],
            axis=0,
        )

        self.roots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.multiplicities = np.ones(10)


class TestRootfinding151(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.5.1 from [KB]
        self.C = Rectangle([-2, 2], [-2, 3])
        self.f = lambda z: exp(3 * z) + 2 * z * cos(z) - 1
        self.df = lambda z: 3 * exp(3 * z) + 2 * cos(z) - 2 * z * sin(z)

        self.roots = np.array(
            [
                -1.84423395326221337491592440,
                0,
                0.5308949302929305274642203840 - 1.331791876751120981651544228j,
                0.5308949302929305274642203840 + 1.331791876751120981651544228j,
            ],
            dtype=complex,
        )
        self.multiplicities = [1, 1, 1, 1]

    def test_rootfinding_b_df(self):
        roots_approx_equal(
            self.C.roots(self.f, self.df, M=2),
            (self.roots, self.multiplicities),
            decimal=12,
        )

    def test_rootfinding_b_f(self):
        roots_approx_equal(
            self.C.roots(self.f, M=2),
            (self.roots, self.multiplicities),
            decimal=12,
        )


class TestRootfinding152(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.5.2 from [KB]
        self.C = Rectangle([-0.5, 5.5], [-0.5, 1.5])
        self.f = lambda z: z**2 * (z - 1) * (z - 2) * (z - 3) * (z - 4) + z * sin(z)
        self.df = (
            lambda z: 2 * z * (3 * z**4 - 25 * z**3 + 70 * z**2 - 75 * z + 24)
            + sin(z)
            + z * cos(z)
        )

        self.roots = [
            0,
            1.18906588973011365517521756,
            1.72843498616506284043592924,
            3.01990732809571222812005354,
            4.03038191606046844562845941,
        ]
        self.multiplicities = [2, 1, 1, 1, 1]


class TestRootfinding153(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        # Ex 1.5.3 from [KB]
        self.C = Rectangle([-1, 3], [-1, 1])
        self.f = lambda z: (z * (z - 2)) ** 2 * (
            exp(2 * z) * cos(z) + z**3 - 1 - sin(z)
        )
        self.df = (
            lambda z: 2 * z * (z - 2) ** 2 * (exp(2 * z) * cos(z) + z**3 - 1 - sin(z))
            + 2 * z**2 * (z - 2) * (exp(2 * z) * cos(z) + z**3 - 1 - sin(z))
            + (z * (z - 2)) ** 2
            * (2 * exp(2 * z) * cos(z) - exp(2 * z) * sin(z) + 3 * z**2 - cos(z))
        )

        self.roots = [
            0,
            2,
            1.66468286974551654134568653,
            -0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
            -0.4607141197289707542294459477 + 0.6254277693477682516688207854j,
        ]
        self.multiplicities = [3, 2, 1, 1, 1]


def test_reevaluation_of_num_roots():
    contour = Circle(0, 2)

    def f(z):
        return (z - 1) * (z - 0.2) ** 2

    roots = [1, 0.2]
    multiplicities = [1, 2]
    roots_approx_equal(
        contour.roots(f, int_abs_tol=10, int_method="romb"),
        (roots, multiplicities),
    )


class TestIntroduction(unittest.TestCase, RootfindingTests, MultiplicityTests):
    def setUp(self):
        self.C = Circle(0, 3)
        self.f = lambda z: (z * (z + 2)) ** 2 * (
            exp(2 * z) * cos(z) - 1 - sin(z) + z**5
        )
        self.df = lambda z: 2 * (exp(2 * z) * cos(z) - 1 - sin(z) + z**5) * (
            z**2 * (z + 2) + (z + 2) ** 2 * z
        ) + (z * (z + 2)) ** 2 * (
            2 * exp(2 * z) * cos(z) - exp(2 * z) * sin(z) - cos(z) + 5 * z**4
        )

        self.roots = [
            0,
            -2,
            2.23755778246706002284084684,
            -0.6511140702635986824274097994 - 0.3904257190882864369857773146j,
            -0.6511140702635986824274097994 + 0.3904257190882864369857773146j,
            0.64857808095387581293067569277 - 1.35662268398824203963215495605j,
            0.64857808095387581293067569277 + 1.35662268398824203963215495605j,
        ]
        self.multiplicities = [3, 2, 1, 1, 1, 1, 1]


def test_annular_combustion():
    A = -0.19435  # noqa: N806
    B = 1000.41  # noqa: N806
    C = 522463  # noqa: N806
    T = 0.005  # noqa: N806

    def f(z):
        return z**2 + A * z + B * exp(-T * z) + C

    def df(z):
        return 2 * z + A - B * T * exp(-T * z)

    rectangle = Rectangle([-15000, 5000], [-15000, 15000])

    import warnings

    warnings.filterwarnings("error")
    roots = rectangle.roots(f, df, root_err_tol=1e-6)
    assert len(roots.roots) == 24


@pytest.mark.parametrize("int_method", ["quad", "romb"])
def test_const_df(int_method):
    def f(z):
        return z - 0.5

    def df(z):
        return 1

    contour = Circle(0, 1)
    roots = contour.roots(f, df, int_method=int_method)

    assert roots.roots == [0.5]
    assert roots.multiplicities == [1]


@pytest.mark.parametrize("int_method", ["quad", "romb"])
def test_df(int_method):
    def f(z):
        return (z - 0.5) ** 2

    def df(z):
        return 2 * (z - 0.5)

    contour = Circle(0, 1)
    roots = contour.roots(f, df, int_method=int_method)

    assert roots.roots == pytest.approx([0.5])
    assert roots.multiplicities == [2]


if __name__ == "__main__":
    unittest.main(verbosity=3)
