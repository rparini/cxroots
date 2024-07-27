import logging
import warnings
from collections.abc import Callable
from math import inf, pi
from typing import overload

import numpy as np
import numpy.typing as npt

from .contour_interface import ContourABC
from .derivative import central_diff
from .types import AnalyticFunc, ComplexScalarOrArray, IntegrationMethod, ScalarOrArray

RombCallback = Callable[[complex, float | None, int], bool | None]


def prod(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: AnalyticFunc | None = None,
    phi: AnalyticFunc | None = None,
    psi: AnalyticFunc | None = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    div_min: int = 3,
    div_max: int = 15,
    int_method: IntegrationMethod = "quad",
    integer_tol: float = inf,
    callback: RombCallback | None = None,
) -> complex:
    r"""
    Compute the symmetric bilinear form used in (1.12) of [KB]_.

    .. math::

        <\phi,\psi> = \frac{1}{2\pi i} \oint_C \phi(z) \psi(z) \frac{f'(z)}{f(z)} dz.

    Parameters
    ----------
    C : :class:`Contour <cxroots.contour.Contour>`
        A contour in the complex plane for.  No roots or poles of f
        should lie on C.
    f : function
        Function of a single variable f(x)
    df : function, optional
        Function of a single variable, df(x), providing the derivative
        of the function f(x) at the point x.  If not provided then df is
        approximated using a finite difference method.
    phi : function, optional
        Function of a single variable phi(x).  If not provided then
        phi(z)=1.
    psi : function, optional
        Function of a single variable psi(x).  If not provided then
        psi(z)=1.
    abs_tol : float, optional
        Absolute error tolerance for integration.
    rel_tol : float, optional
        Relative error tolerance for integration.
    div_min : int, optional
        Only used if int_method='romb'. Minimum number of divisions before
        the Romberg integration routine is allowed to exit.
    div_max : int, optional
        Only used if int_method='romb'.  The maximum number of divisions
        before the Romberg integration routine of a path exits.
    int_method : {'quad', 'romb'}, optional
        If 'quad' then scipy.integrate.quad is used to perform the
        integral.  If 'romb' then Romberg integraion, using
        scipy.integrate.romb, is performed instead.
    integer_tol : float, optional
        Only used when int_method is 'romb'.  The integration routine will
        not exit unless the result is within integer_tol of an integer.
        This is useful when computing the number of roots in a contour,
        which must be an integer.  By default integer_tol is inf.
    callback : function, optional
        Only used when int_method is 'romb'.  A function that at each
        step in the iteration is passed the current approximation for
        the integral, the estimated error of that approximation and the
        number of iterations.  If the return of callback evaluates to
        True then the integration will end.

    Returns
    -------
    complex
        The value of the integral <phi, psi>.
    float
        An estimate of the error for the integration.

    References
    ----------
    .. [KB] "Computing the zeros of analytic functions" by Peter Kravanja,
        Marc Van Barel, Springer 2000
    """
    if int_method == "romb":
        return _romb_prod(
            C,
            f,
            df,
            phi,
            psi,
            abs_tol,
            rel_tol,
            div_min,
            div_max,
            integer_tol,
            callback,
        )
    elif int_method == "quad":
        return _quad_prod(C, f, df, phi, psi, abs_tol, rel_tol)
    else:
        raise ValueError("int_method must be either 'romb' or 'quad'")


def _romb_prod(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: AnalyticFunc | None = None,
    phi: AnalyticFunc | None = None,
    psi: AnalyticFunc | None = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    div_min: int = 3,
    div_max: int = 15,
    integer_tol: float = inf,
    callback: RombCallback | None = None,
) -> complex:
    logger = logging.getLogger(__name__)
    k = 0
    I = []  # List of approximations to the integral # noqa: E741 N806

    while k < div_max and (
        len(I) < div_min
        or (abs(I[-2] - I[-1]) > abs_tol and abs(I[-2] - I[-1]) > rel_tol * abs(I[-1]))
        or (abs(I[-3] - I[-2]) > abs_tol and abs(I[-3] - I[-2]) > rel_tol * abs(I[-2]))
        or abs(int(round(I[-1].real)) - I[-1].real) > integer_tol
        or abs(I[-1].imag) > integer_tol
    ):
        k += 1
        integral = C.trap_product(k, f, df, phi, psi)
        I.append(integral)
        if k > 1:
            logger.debug(f"Iteration={k}, integral={I[-1]}, err={I[-2] - I[-1]}")
        else:
            logger.debug(f"Iteration={k}, integral={I[-1]}")

        if callback is not None:
            err = abs(I[-2] - I[-1]) if k > 1 else None
            if callback(I[-1], err, k):
                break

    return I[-1]


def _quad_prod(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: AnalyticFunc | None = None,
    phi: AnalyticFunc | None = None,
    psi: AnalyticFunc | None = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
) -> complex:
    if df is None:
        df = central_diff(f)

    def one(z: ScalarOrArray) -> int:
        return 1

    if phi is None:
        phi = one
    if psi is None:
        psi = one

    @overload
    def integrand_func(z: complex | float) -> complex: ...

    @overload
    def integrand_func(
        z: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128] | complex: ...

    def integrand_func(z: ScalarOrArray) -> ComplexScalarOrArray:
        return phi(z) * psi(z) * (df(z) / f(z)) / (2j * pi)

    return C.integrate(
        integrand_func, abs_tol=abs_tol, rel_tol=rel_tol, int_method="quad"
    )


class RootError(RuntimeError):
    pass


def count_roots(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: AnalyticFunc | None = None,
    int_abs_tol: float = 0.07,
    integer_tol: float = 0.1,
    div_min: int = 3,
    div_max: int = 15,
    int_method: IntegrationMethod = "quad",
) -> int:
    r"""
    For a function of one complex variable, f(z), which is analytic in
    and within the contour C, return the number of zeros (counting
    multiplicities) within the contour, N, using Cauchy's argument
    principle,

    .. math::

        N = \frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

    If df(z), the derivative of f(z), is provided then the above
    integral is computed directly.  Otherwise the derivative is
    approximated using a finite difference method.

    The number of roots is taken to be the closest integer to the
    computed value of the integral and the result is only accepted
    if the integral is within integer_tol of the closest integer.

    Parameters
    ----------
    C : :class:`Contour <cxroots.contour.Contour>`
        The contour which encloses the roots of f(z) that are to be
        counted.
    f : function
        Function of a single variable f(z).
    df : function, optional
        Function of a single complex variable, df(z), providing the
        derivative of the function f(z) at the point z.  If not
        provided, df will be approximated using a finite difference
        method.
    int_abs_tol : float, optional
        Required absolute error tolerance for the contour integration.
        Since the Cauchy integral must be an integer it is only
        necessary to distinguish which integer the integral is
        converging towards.  Therefore, int_abs_tol can be fairly large.
    integer_tol : float, optional
        The evaluation of the Cauchy integral will be accepted if its
        value is within integer_tol of the closest integer.
    div_min : int, optional
        Only used if int_method='romb'. Minimum number of divisions
        before the Romberg integration routine is allowed to exit.
    div_max : int, optional
        Only used if int_method='romb'. The maximum number of divisions
        before the Romberg integration routine of a path exits.
    int_method : {'quad', 'romb'}, optional
        If 'quad' then scipy.integrate.quad is used to perform the
        integral.  If 'romb' then Romberg integraion, using
        scipy.integrate.romb, is performed instead.

    Returns
    -------
    int
        The number of zeros of f (counting multiplicities) which lie
        within the contour C.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Computing number of roots within " + str(C))

    with warnings.catch_warnings():
        # ignore warnings and catch if integral is NaN later
        warnings.simplefilter("ignore")
        integral = prod(
            C,
            f,
            df,
            abs_tol=int_abs_tol,
            rel_tol=0,
            div_min=div_min,
            div_max=div_max,
            int_method=int_method,
            integer_tol=integer_tol,
        )

    logger.debug(f"Integral for number of roots = {integral}")

    if np.isnan(integral):
        raise RootError(
            "Result of integral is an invalid value. "
            "Most likely because of a divide by zero error."
        )

    elif (
        abs(int(round(integral.real)) - integral.real) < integer_tol
        and abs(integral.imag) < integer_tol
    ):
        # integral is sufficiently close to an integer
        num_zeros = int(round(integral.real))
        logger.info(
            "Counted "
            + str(num_zeros)
            + " roots (including multiplicities) within "
            + str(C)
        )

        return num_zeros

    else:
        raise RootError("The number of enclosed roots has not converged to an integer")
