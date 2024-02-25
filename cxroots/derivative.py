from math import factorial, pi
from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt

from .contour import Contour
from .types import AnalyticFunc, ComplexScalarOrArray, ScalarOrArray


@np.vectorize
def cx_derivative(
    f: AnalyticFunc,
    z0: complex,
    n: int = 1,
    contour: Optional[Contour] = None,
    integration_abs_tol: float = 1.49e-08,
) -> complex:
    r"""
    Compute the derivaive of an analytic function using Cauchy's
    Integral Formula for Derivatives.

    .. math::

        f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z-z_0)^{n+1}} dz

    Parameters
    ----------
    f : function
        Function of a single variable f(x).
    z0 : complex
        Point to evaluate the derivative at.
    n : int
        The order of the derivative to evaluate.
    contour : :class:`Contour <cxroots.contour.Contour>`, optional
        The contour, C, in the complex plane which encloses the point z0.
        By default the contour is the circle |z-z_0|=1e-3.
    integration_abs_tol : float, optional
        The absolute tolerance required of the integration routine.

    Returns
    -------
    f^{(n)}(z0) : complex
        The nth derivative of f evaluated at z0
    """
    from .contours.circle import Circle

    if contour is None:
        contour = Circle(z0, 1e-3)

    def integrand(z):
        return f(z) / (z - z0) ** (n + 1)

    integral = contour.integrate(integrand, abs_tol=integration_abs_tol)
    return integral * factorial(n) / (2j * pi)


def central_diff(
    f: AnalyticFunc,
) -> AnalyticFunc:
    h = 5e-6

    @overload
    def df(
        z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> ComplexScalarOrArray:
        ...

    @overload
    def df(z: Union[complex, float]) -> complex:
        ...

    def df(z: ScalarOrArray) -> ComplexScalarOrArray:
        return (f(z + h) - f(z - h)) / (2 * h)

    return df
