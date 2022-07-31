from __future__ import division
from numpy import pi

from ..Contour import Contour
from ..Paths import ComplexArc
from .Annulus import Annulus


class Circle(Contour):
    """
    A positively oriented circle in the complex plane.

    Parameters
    ----------
    center : complex
        The center of the circle.
    radius : float
        The radius of the circle.

    Examples
    --------
    .. plot::
        :include-source:

        from cxroots import Circle
        circle = Circle(center=1, radius=0.5)
        circle.show()
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.axis_name = "r"

        segments = [ComplexArc(center, radius, 0, 2 * pi)]
        super(Circle, self).__init__(segments)

    def __str__(self):
        return (
            f"Circle: center={self.center.real:.3f}{self.center.imag:+.3f}i, "
            f"radius={self.radius:.3f}"
        )

    def contains(self, z):
        """Returns True if the point z lies within the contour, False if otherwise"""
        return abs(z - self.center) < self.radius

    @property
    def central_point(self):
        return self.center

    @property
    def area(self):
        return pi * self.radius**2

    def subdivide(self, axis="r", division_factor=0.5):
        """
        Subdivide the contour

        Parameters
        ----------
        axis : str, can only be 'r'
            The axis along which the line subdividing the contour is a constant.
        division_factor : float in range (0,1), optional
            Determines the point along 'axis' at which the line dividing the box is
            placed

        Returns
        -------
        box1 : Annulus
            With inner radius determined by the division_factor and outer radius equal
            to that of the original circle
        box2 : Circle
            With radius equal to the inner radius of box1
        """
        if axis == "r" or self.axis_name[axis] == "r":
            box1 = Annulus(self.center, [self.radius * division_factor, self.radius])
            box2 = Circle(self.center, self.radius * division_factor)
            box1.segments[0] = self.segments[0]
            box1.segments[1]._reversePath = box2.segments[0]
            box2.segments[0]._reversePath = box1.segments[1]

        for box in [box1, box2]:
            box._created_by_subdivision_axis = axis
            box._parentBox = self
            self._childBoxes = [box1, box2]

        return box1, box2
