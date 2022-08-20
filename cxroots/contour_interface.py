from abc import ABC, abstractmethod
from typing import Generator, Optional, Sequence, Tuple

from .types import AnalyticFunc, Color, IntegrationMethod


class ContourABC(ABC):
    @property
    @abstractmethod
    def central_point(self) -> complex:
        """A central point that lies within the contour"""
        ...

    @property
    @abstractmethod
    def area(self) -> float:
        """The area of the contour in the complex plane"""
        ...

    @abstractmethod
    def contains(self, z: complex) -> bool:
        """True if the point z is within the contour, false otherwise"""
        ...

    @abstractmethod
    def distance(self, z: complex) -> float:
        """
        The distance from the point z in the complex plane to the
        nearest point on the contour.
        """
        ...

    @abstractmethod
    def trap_product(
        self,
        k: int,
        f: AnalyticFunc,
        df: Optional[AnalyticFunc] = None,
        phi: Optional[AnalyticFunc] = None,
        psi: Optional[AnalyticFunc] = None,
    ) -> complex:
        r"""
        Use Romberg integration to estimate the symmetric bilinear form used in
        (1.12) of [KB]_ using 2**k+1 samples

        .. math::

            <\phi,\psi> = \frac{1}{2\pi i} \oint_C \phi(z)\psi(z)\frac{f'(z)}{f(z)} dz
        """
        ...

    @abstractmethod
    def integrate(
        self,
        f: AnalyticFunc,
        abs_tol: float = 1.49e-08,
        rel_tol: float = 1.49e-08,
        div_max: int = 15,
        int_method: IntegrationMethod = "quad",
    ) -> complex:
        r"""
        Integrate the function f along the contour C

        .. math::

            \oint_C f(z) dz
        """
        ...

    @abstractmethod
    def count_roots(
        self,
        f: AnalyticFunc,
        df: Optional[AnalyticFunc] = None,
        int_abs_tol: float = 0.07,
        integer_tol: float = 0.1,
        div_min: int = 3,
        div_max: int = 15,
        df_approx_order: int = 2,
        int_method: IntegrationMethod = "quad",
    ) -> int:
        """
        For a function of one complex variable, f(z), which is analytic in
        and within the contour C, return the number of zeros (counting
        multiplicities) within the contour
        """
        ...

    @abstractmethod
    def subdivisions(
        self, axis: str = "alternating"
    ) -> Generator[Tuple["ContourABC", ...], None, None]:
        """A generator for possible subdivisions of the contour"""
        ...

    @property
    @abstractmethod
    def parent(self) -> Optional["ContourABC"]:
        """The contour that this contour was created from as part of subdivision"""
        ...

    @property
    @abstractmethod
    def children(self) -> Optional[Sequence["ContourABC"]]:
        """The contours that were created from this contour during subdivision"""
        ...

    @abstractmethod
    def plot(
        self, num_points: int = 100, linecolor: Color = "C0", linestyle: str = "-"
    ) -> None:
        """
        Uses matplotlib to plot, but not show, the contour as a 2D plot in
        the Complex plane.
        """
        ...

    @abstractmethod
    def size_plot(self) -> None:
        """Adjust the plot axes limits to nicely frame the contour"""
        ...
