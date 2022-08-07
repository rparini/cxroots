import unittest

from numpy import pi

from cxroots import Annulus, AnnulusSector, Circle, Rectangle
from cxroots.tests.approx_equal import roots_approx_equal


class TestRootfindingContours(unittest.TestCase):
    def setUp(self):
        self.roots = roots = [0, -1.234, 1 + 1j, 1 - 1j, 2.345]
        self.multiplicities = [1, 1, 1, 1, 1]
        self.f = (
            lambda z: (z - roots[0])
            * (z - roots[1])
            * (z - roots[2])
            * (z - roots[3])
            * (z - roots[4])
        )
        self.df = (
            lambda z: (z - roots[1]) * (z - roots[2]) * (z - roots[3]) * (z - roots[4])
            + (z - roots[0]) * (z - roots[2]) * (z - roots[3]) * (z - roots[4])
            + (z - roots[0]) * (z - roots[1]) * (z - roots[3]) * (z - roots[4])
            + (z - roots[0]) * (z - roots[1]) * (z - roots[2]) * (z - roots[4])
            + (z - roots[0]) * (z - roots[1]) * (z - roots[2]) * (z - roots[3])
        )

        self.Circle = Circle(0, 3)
        self.Rectangle = Rectangle([-2, 2], [-2, 2])
        self.halfAnnulus = AnnulusSector(0, [0.5, 3], [-pi / 2, pi / 2])
        self.Annulus = Annulus(0, [1, 2])

    def test_rootfinding_circle_fdf(self):
        roots_approx_equal(
            self.Circle.roots(self.f, self.df),
            (self.roots, self.multiplicities),
            decimal=7,
        )

    def test_rootfinding_circle_f(self):
        roots_approx_equal(
            self.Circle.roots(self.f, self.df),
            (self.roots, self.multiplicities),
            decimal=7,
        )

    def test_rootfinding_rectangle_fdf(self):
        roots_approx_equal(
            self.Rectangle.roots(self.f, self.df),
            (self.roots[:-1], self.multiplicities[:-1]),
            decimal=7,
        )

    def test_rootfinding_rectangle_f(self):
        roots_approx_equal(
            self.Rectangle.roots(self.f, self.df),
            (self.roots[:-1], self.multiplicities[:-1]),
            decimal=7,
        )

    def test_rootfinding_half_annulus_fdf(self):
        roots_approx_equal(
            self.halfAnnulus.roots(self.f, self.df),
            (self.roots[2:], self.multiplicities[2:]),
            decimal=7,
        )

    def test_rootfinding_half_annulus_f(self):
        roots_approx_equal(
            self.halfAnnulus.roots(self.f, self.df),
            (self.roots[2:], self.multiplicities[2:]),
            decimal=7,
        )

    def test_rootfinding_annulus_fdf(self):
        roots_approx_equal(
            self.Annulus.roots(self.f, self.df),
            (self.roots[1:-1], self.multiplicities[1:-1]),
            decimal=7,
        )

    def test_rootfinding_annulus_f(self):
        roots_approx_equal(
            self.Annulus.roots(self.f, self.df),
            (self.roots[1:-1], self.multiplicities[1:-1]),
            decimal=7,
        )
