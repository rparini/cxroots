import logging
import warnings
from math import inf, pi
from typing import Callable, Optional, Union, overload

import numdifftools
import numpy as np
import numpy.typing as npt

from .contour_interface import ContourABC
from .types import AnalyticFunc, ComplexScalarOrArray, IntegrationMethod, ScalarOrArray

RombCallback = Callable[[complex, Optional[float], int], Optional[bool]]


def prod(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    phi: Optional[AnalyticFunc] = None,
    psi: Optional[AnalyticFunc] = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    div_min: int = 3,
    div_max: int = 15,
    df_approx_order: int = 2,
    int_method: IntegrationMethod = "quad",
    integer_tol: float = inf,
    callback: Optional[RombCallback] = None,
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
    df_approx_order : int, optional
        Only used if df=None and int_method='quad'.  Must be even.  The
        argument order=df_approx_order is passed to numdifftools.Derivative and is the
        order of the error term in the Taylor approximation.
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
        return _quad_prod(C, f, df, phi, psi, abs_tol, rel_tol, df_approx_order)
    else:
        raise ValueError("int_method must be either 'romb' or 'quad'")


def _romb_prod(
    C: ContourABC,  # noqa: N803
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    phi: Optional[AnalyticFunc] = None,
    psi: Optional[AnalyticFunc] = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    div_min: int = 3,
    div_max: int = 15,
    integer_tol: float = inf,
    callback: Optional[RombCallback] = None,
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
    df: Optional[AnalyticFunc] = None,
    phi: Optional[AnalyticFunc] = None,
    psi: Optional[AnalyticFunc] = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    df_approx_order: int = 2,
) -> complex:
    if df is None:
        df = numdifftools.Derivative(f, order=df_approx_order)  # type: ignore
        # type checker needs this reassurance for some reason
        assert df is not None  # nosec B101

        # Using scipy.misc.derivative leads to some roots being missed in tests
        # df = lambda z: scipy.misc.derivative(f, z, dx=1.49e-8, n=1, order=3)

        # Too slow
        # import numdifftools.fornberg as ndf
        # ndf.derivative returns an array [f, f', f'', ...]
        # df = np.vectorize(lambda z: ndf.derivative(f, z, n=1)[1])

    def one(z: ScalarOrArray) -> int:
        return 1

    if phi is None:
        phi = one
    if psi is None:
        psi = one

    @overload
    def integrand_func(z: Union[complex, float]) -> complex:
        ...

    @overload
    def integrand_func(
        z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> Union[npt.NDArray[np.complex_], complex]:
        ...

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
    df: Optional[AnalyticFunc] = None,
    int_abs_tol: float = 0.07,
    integer_tol: float = 0.1,
    div_min: int = 3,
    div_max: int = 15,
    df_approx_order: int = 2,
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
    df_approx_order : int, optional
        Only used if df=None and int_method='quad'.  The argument order=df_approx_order
        is passed to numdifftools.Derivative and is the order of the
        error term in the Taylor approximation.  df_approx_order must be even.
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
            df_approx_order=df_approx_order,
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
