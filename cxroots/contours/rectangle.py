from typing import Literal, Tuple

from ..contour import Contour
from ..paths import ComplexLine


class Rectangle(Contour):
    """
    A positively oriented rectangle in the complex plane.

    Parameters
    ----------
    x_range : tuple
            Tuple of length two giving the range of the rectangle along the
            real axis.
    y_range : tuple
            Tuple of length two giving the range of the rectangle along the
            imaginary axis.

    Examples
    --------
    .. plot::
            :include-source:

            from cxroots import Rectangle
            rect = Rectangle(x_range=(-2, 2), y_range=(-1, 1))
            rect.show()
    """

    axis_names = ("x", "y")

    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        self.x_range = x_range
        self.y_range = y_range

        self.z1 = z1 = self.x_range[0] + 1j * self.y_range[0]
        self.z2 = z2 = self.x_range[1] + 1j * self.y_range[0]
        self.z3 = z3 = self.x_range[1] + 1j * self.y_range[1]
        self.z4 = z4 = self.x_range[0] + 1j * self.y_range[1]

        segments = [
            ComplexLine(z1, z2),
            ComplexLine(z2, z3),
            ComplexLine(z3, z4),
            ComplexLine(z4, z1),
        ]
        super(Rectangle, self).__init__(segments)

    def __str__(self):
        return (
            "Rectangle: vertices = "
            f"{self.z1.real:.3f}{self.z1.imag:+.3f}i, "
            f"{self.z2.real:.3f}{self.z2.imag:+.3f}i, "
            f"{self.z3.real:.3f}{self.z3.imag:+.3f}i, "
            f"{self.z4.real:.3f}{self.z4.imag:+.3f}i"
        )

    @property
    def central_point(self) -> complex:
        # get the central point within the contour
        x = (self.x_range[0] + self.x_range[1]) / 2
        y = (self.y_range[0] + self.y_range[1]) / 2
        return x + 1j * y

    @property
    def area(self) -> float:
        return (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])

    def contains(self, z: complex) -> bool:
        """Returns True if the point z lies within the contour, False if otherwise"""
        return (
            self.x_range[0] < z.real < self.x_range[1]
            and self.y_range[0] < z.imag < self.y_range[1]
        )

    def subdivide(
        self, axis: Literal["x", "y"], division_factor: float = 0.5
    ) -> Tuple["Rectangle", "Rectangle"]:
        """
        Subdivide the contour

        Parameters
        ----------
        axis : str, can be either 'x' or 'y'
            The axis along which the line subdividing the contour is a constant.
        division_factor : float in range (0,1), optional
            Determines the point along 'axis' at which the line dividing the contour
            is placed.

        Returns
        -------
        box1 : Rectangle
            If axis is 'x' then box1 has the same y_range and minimum value of x_range
            as the original Rectangle but the maximum x_range is determined by the
            division_factor.
            If axis is 'y' then box1 has the same x_range and minimum value of y_range
            as the original Rectangle but the maximum y_range is determined by the
            division_factor.
        box2 : Rectangle
            If axis is 'x' then box2 has the same y_range and maximum value of x_range
            as the original Rectangle but the minimum x_range is equal to the maximum
            x_range of box1.
            If axis is 'x' then box2 has the same x_range and maximum value of y_range
            as the original Rectangle but the minimum y_range is equal to the maximum
            y_range of box1.
        """
        if axis == "x":
            midpoint = self.x_range[0] + division_factor * (
                self.x_range[1] - self.x_range[0]
            )
            box1 = Rectangle((self.x_range[0], midpoint), self.y_range)
            box2 = Rectangle((midpoint, self.x_range[1]), self.y_range)

            box1.segments[3] = self.segments[3]
            box2.segments[1] = self.segments[1]
            box1.segments[1]._reverse_path = box2.segments[3]
            box2.segments[3]._reverse_path = box1.segments[1]

        elif axis == "y":
            midpoint = self.y_range[0] + division_factor * (
                self.y_range[1] - self.y_range[0]
            )
            box1 = Rectangle(self.x_range, (self.y_range[0], midpoint))
            box2 = Rectangle(self.x_range, (midpoint, self.y_range[1]))

            box1.segments[0] = self.segments[0]
            box2.segments[2] = self.segments[2]
            box1.segments[2]._reverse_path = box2.segments[0]
            box2.segments[0]._reverse_path = box1.segments[2]

        else:
            raise ValueError("axis must be 'x' or 'y'")

        for box in [box1, box2]:
            box._created_by_subdivision_axis = axis
            box._parent = self
        self._children = [box1, box2]

        return box1, box2
