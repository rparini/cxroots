from typing import Literal, Tuple

from numpy import pi

from ..contour import Contour
from ..paths import ComplexArc
from .annulus import Annulus


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

    axis_names = tuple("r")

    def __init__(self, center: complex, radius: float):
        self.center = center
        self.radius = radius

        segments = [ComplexArc(center, radius, 0, 2 * pi)]
        super(Circle, self).__init__(segments)

    def __str__(self):
        return (
            f"Circle: center={self.center.real:.3f}{self.center.imag:+.3f}i, "
            f"radius={self.radius:.3f}"
        )

    def contains(self, z: complex) -> bool:
        """Returns True if the point z lies within the contour, False if otherwise"""
        return abs(z - self.center) < self.radius

    @property
    def central_point(self) -> complex:
        return self.center

    @property
    def area(self) -> float:
        return pi * self.radius**2

    def subdivide(
        self, axis: Literal["r"] = "r", division_factor: float = 0.5
    ) -> Tuple[Annulus, "Circle"]:
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
        if axis == "r":
            box1 = Annulus(self.center, (self.radius * division_factor, self.radius))
            box2 = Circle(self.center, self.radius * division_factor)
            box1.segments[0] = self.segments[0]
            box1.segments[1]._reverse_path = box2.segments[0]
            box2.segments[0]._reverse_path = box1.segments[1]

        else:
            raise ValueError("axis must be 'r'")

        for box in [box1, box2]:
            box._created_by_subdivision_axis = axis
            box._parent = self
        self._children = [box1, box2]

        return box1, box2
