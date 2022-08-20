from typing import Literal, Tuple, Union, overload

from numpy import pi

from ..contour import Contour
from ..paths import ComplexArc
from .annulus_sector import AnnulusSector


class Annulus(Contour):
    """
    An annulus in the complex plane with the outer circle positively oriented
    and the inner circle negatively oriented.

    Parameters
    ----------
    center : complex
            The center of the annulus in the complex plane.
    radii : tuple
            A tuple of length two of the form (inner_radius, outer_radius).

    Examples
    --------
    .. plot::
            :include-source:

            from cxroots import Annulus
            annulus = Annulus(center=0, radii=(0.5,0.75))
            annulus.show()
    """

    axis_names = ("r", "phi")

    def __init__(self, center: complex, radii: Tuple[float, float]):
        self.center = center
        self.radii = radii

        segments = [
            ComplexArc(center, radii[1], 0, 2 * pi),
            ComplexArc(center, radii[0], 0, -2 * pi),
        ]
        super(Annulus, self).__init__(segments)

    def __str__(self):
        return (
            f"Annulus: center={self.center.real:.3f}{self.center.imag:+.3f}i, "
            f"inner radius={self.radii[0]:.3f}, outer radius={self.radii[1]:.3f}"
        )

    @property
    def central_point(self) -> complex:
        # get a central point within the contour
        r = (self.radii[0] + self.radii[1]) / 2
        return r

    @property
    def area(self) -> float:
        return pi * (self.radii[1] ** 2 - self.radii[0] ** 2)

    def contains(self, z: complex) -> bool:
        """Returns True if the point z lies within the contour, False if otherwise"""
        return self.radii[0] < abs(z - self.center) < self.radii[1]

    @overload
    def subdivide(
        self, axis: Literal["r"], division_factor: float
    ) -> Tuple["Annulus", "Annulus"]:
        ...

    @overload
    def subdivide(
        self, axis: Literal["phi"], division_factor: float
    ) -> Tuple[AnnulusSector, AnnulusSector]:
        ...

    def subdivide(
        self, axis: Literal["r", "phi"], division_factor: float = 0.5
    ) -> Union[Tuple[AnnulusSector, AnnulusSector], Tuple["Annulus", "Annulus"]]:
        """
        Subdivide the contour

        Parameters
        ----------
        axis : str, can be either 'r' or 'phi'
            The axis along which the line subdividing the contour is a constant.
        division_factor : float in range (0,1), optional
            Determines the point along 'axis' at which the line dividing the box is
            placed

        Returns
        -------
        boxes : list of contours
            Two annuluses if axis is 'r'.
            Two half-annuluses oriented according to division_factor if axis is 'phi'.
        """
        if axis == "r":
            midpoint = self.radii[0] + division_factor * (self.radii[1] - self.radii[0])
            boxes = (
                Annulus(self.center, (self.radii[0], midpoint)),
                Annulus(self.center, (midpoint, self.radii[1])),
            )

            boxes[0].segments[1] = self.segments[1]
            boxes[1].segments[0] = self.segments[0]
            boxes[0].segments[0]._reverse_path = boxes[1].segments[1]
            boxes[1].segments[1]._reverse_path = boxes[0].segments[0]

        elif axis == "phi":
            # Subdividing into two radial boxes rather than one to
            # ensure that an error is raised if one of the new paths
            # is too close to a root
            # XXX: introduce another parameter for phi1

            phi0 = 2 * pi * division_factor
            phi1 = phi0 + pi
            boxes = (
                AnnulusSector(self.center, self.radii, (phi0, phi1)),
                AnnulusSector(self.center, self.radii, (phi1, phi0)),
            )

            boxes[0].segments[0]._reverse_path = boxes[1].segments[2]
            boxes[1].segments[2]._reverse_path = boxes[0].segments[0]
            boxes[0].segments[2]._reverse_path = boxes[1].segments[0]
            boxes[1].segments[0]._reverse_path = boxes[0].segments[2]

        else:
            raise ValueError("axis must be 'r' or 'phi'")

        for box in boxes:
            box._created_by_subdivision_axis = axis
            box._parent = self
        self._children = boxes

        return boxes
