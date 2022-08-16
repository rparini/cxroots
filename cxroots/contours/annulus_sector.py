from math import pi
from typing import Literal, Tuple

import numpy as np

from ..contour import Contour
from ..paths import ComplexArc, ComplexLine


class AnnulusSector(Contour):
    """
    A sector of an annulus in the complex plane.

    Parameters
    ----------
    center : complex
        The center of the annulus sector.
    radii : tuple
        Tuple of length two of the form (inner_radius, outer_radius)
    phi_range : tuple
        Tuple of length two of the form (phi0, phi1).
        The segment of the contour containing inner and outer circular
        arcs will be joined, counter clockwise from phi0 to phi1.

    Examples
    --------
    .. plot::
        :include-source:

        from numpy import pi
        from cxroots import AnnulusSector
        annulusSector = AnnulusSector(
            center=0.2, radii=(0.5, 1.25), phi_range=(-pi/4, pi/4)
        )
        annulusSector.show()

    .. plot::
        :include-source:

        from numpy import pi
        from cxroots import AnnulusSector
        annulusSector = AnnulusSector(
            center=0.2, radii=(0.5, 1.25), phi_range=(pi/4, -pi/4)
        )
        annulusSector.show()
    """

    axis_names = ("r", "phi")

    def __init__(
        self,
        center: complex,
        radii: Tuple[float, float],
        phi_range: Tuple[float, float],
    ):
        self.center = center

        if phi_range[0] > phi_range[1]:
            phi_range = (phi_range[0], phi_range[1] + 2 * pi)

        phi0, phi1 = self.phi_range = phi_range

        # r > 0
        r0, r1 = self.radii = radii
        if r0 < 0 or r1 <= 0:
            raise ValueError("Radius > 0")

        # verticies [[radius0,phi0],[radius0,phi1],[radius1,phi1],[radius0,phi1]]
        self.z1 = z1 = center + r0 * np.exp(1j * phi0)
        self.z2 = z2 = center + r1 * np.exp(1j * phi0)
        self.z3 = z3 = center + r1 * np.exp(1j * phi1)
        self.z4 = z4 = center + r0 * np.exp(1j * phi1)

        segments = [
            ComplexLine(z1, z2),
            ComplexArc(center, r1, phi0, phi1 - phi0),
            ComplexLine(z3, z4),
            ComplexArc(center, r0, phi1, phi0 - phi1),
        ]

        super(AnnulusSector, self).__init__(segments)

    def __str__(self):
        return (
            f"Annulus sector: center={self.center.real:.3f}{self.center.imag:+.3f}i, "
            f"r0={self.radii[0]:.3f}, r1={self.radii[1]:.3f}, "
            f"phi0={self.phi_range[0]:.3f}, phi1={self.phi_range[1]:.3f}"
        )

    @property
    def central_point(self) -> complex:
        # get the central point within the contour
        r = (self.radii[0] + self.radii[1]) / 2
        phi = (self.phi_range[0] + self.phi_range[1]) / 2
        return r * np.exp(1j * phi)

    @property
    def area(self) -> float:
        return (
            (self.radii[1] ** 2 - self.radii[0] ** 2)
            * abs(self.phi_range[1] - self.phi_range[0])
            % (2 * pi)
            / 2
        )

    def contains(self, z: complex) -> bool:
        """Returns True if the point z lies within the contour, False if otherwise"""
        angle = float(np.angle(z - self.center)) % (2 * pi)  # np.angle maps to [-pi,pi]
        radius_correct = self.radii[0] < abs(z - self.center) < self.radii[1]

        phi = np.mod(self.phi_range, 2 * pi)
        if phi[0] > phi[1]:
            angle_correct = (phi[0] < angle <= 2 * pi) or (0 <= angle < phi[1])
        else:
            angle_correct = phi[0] < angle < phi[1]

        return radius_correct and angle_correct

    def subdivide(
        self, axis: Literal["r", "phi"], division_factor: float = 0.5
    ) -> Tuple["AnnulusSector", "AnnulusSector"]:
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
        box1 : AnnulusSector
            If axis is 'r' then phi_range and the inner radius is the same as original
            AnnulusSector with the outer radius determined by the division_factor.
            If axis is 'phi' then the radii and phi_range[0] is the same as the original
            AnnulusSector with phi_range[1] determined by the division_factor.
        box2 : AnnulusSector
            If axis is 'r' then phi_range and the outer radius is the same as original
            AnnulusSector with the inner radius determined equal to the outer radius
            of box1. If axis is 'phi' then the radii and phi_range[1] is the same as
            the original AnnulusSector with phi_range[0] equal to phi_range[1] of box1.
        """
        r0, r1 = self.radii
        phi0, phi1 = self.phi_range

        if axis == "r":
            division_point = r0 + division_factor * (r1 - r0)
            box1 = AnnulusSector(self.center, (r0, division_point), self.phi_range)
            box2 = AnnulusSector(self.center, (division_point, r1), self.phi_range)

            # reuse line segments from original box where possible
            # this allows the cached integrals to be used
            box1.segments[3] = self.segments[3]
            box2.segments[1] = self.segments[1]
            box1.segments[1]._reverse_path = box2.segments[3]
            box2.segments[3]._reverse_path = box1.segments[1]

        elif axis == "phi":
            division_point = phi0 + division_factor * (phi1 - phi0)
            box1 = AnnulusSector(self.center, self.radii, (phi0, division_point))
            box2 = AnnulusSector(self.center, self.radii, (division_point, phi1))

            box1.segments[0] = self.segments[0]
            box2.segments[2] = self.segments[2]
            box1.segments[2]._reverse_path = box2.segments[0]
            box2.segments[0]._reverse_path = box1.segments[2]

        else:
            raise ValueError("axis must be 'r' or 'phi'")

        for box in [box1, box2]:
            box._created_by_subdivision_axis = axis
            box._parent = self
        self._children = [box1, box2]

        return box1, box2
