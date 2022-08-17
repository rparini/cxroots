from abc import ABC, abstractmethod
from typing import Optional

from .types import AnalyticFunc, IntegrationMethod


class ContourABC(ABC):
    @property
    @abstractmethod
    def central_point(self) -> complex:
        """
        A central point that lies within the contour
        """
        ...

    @property
    @abstractmethod
    def area(self) -> float:
        """
        The area of the contour in the complex plane
        """
        ...

    @abstractmethod
    def contains(self, z: complex) -> bool:
        """
        True if the point z is within the contour, false otherwise
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
