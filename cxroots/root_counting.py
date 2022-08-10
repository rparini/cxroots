import logging
import warnings
from math import inf, pi

import numdifftools
import numpy as np
import scipy.integrate

from .util import integrate_quad_complex


def prod(
    C,  # noqa: N803
    f,
    df=None,
    phi=None,
    psi=None,
    abs_tol=1.49e-08,
    rel_tol=1.49e-08,
    div_min=3,
    div_max=15,
    df_approx_order=2,
    int_method="quad",
    integer_tol=inf,
    callback=None,
):
    r"""
    Compute the symmetric bilinear form used in (1.12) of [KB]_.

    .. math::

        <\phi,\psi> = \frac{1}{2\pi i} \oint_C \phi(z) \psi(z) \frac{f'(z)}{f(z)} dz.

    Parameters
    ----------
    C : :class:`Contour <cxroots.Contour.Contour>`
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
    C,  # noqa: N803
    f,
    df=None,
    phi=None,
    psi=None,
    abs_tol=1.49e-08,
    rel_tol=1.49e-08,
    div_min=3,
    div_max=15,
    integer_tol=inf,
    callback=None,
):
    logger = logging.getLogger(__name__)
    N = 1  # noqa: N806
    k = 0
    I = []  # List of approximations to the integral # noqa: E741 N806

    while k < div_max and (
        len(I) < div_min
        or (abs(I[-2] - I[-1]) > abs_tol and abs(I[-2] - I[-1]) > rel_tol * abs(I[-1]))
        or (abs(I[-3] - I[-2]) > abs_tol and abs(I[-3] - I[-2]) > rel_tol * abs(I[-2]))
        or abs(int(round(I[-1].real)) - I[-1].real) > integer_tol
        or abs(I[-1].imag) > integer_tol
    ):
        N *= 2
        t = np.linspace(0, 1, N + 1)
        k += 1
        dt = t[1] - t[0]

        integrals = []
        for segment in C.segments:
            # compute/retrieve function evaluations
            f_val = segment.trap_values(f, k)

            if df is None:
                # approximate df/dz with finite difference
                dfdt = np.gradient(f_val, dt)
                df_val = dfdt / segment.dzdt(t)
            else:
                df_val = segment.trap_values(df, k)

            segment_integrand = df_val / f_val * segment.dzdt(t)
            if phi is not None:
                segment_integrand = segment.trap_values(phi, k) * segment_integrand
            if psi is not None:
                segment_integrand = segment.trap_values(psi, k) * segment_integrand

            segment_integral = scipy.integrate.romb(
                segment_integrand, dx=dt, axis=-1
            ) / (2j * pi)
            integrals.append(segment_integral)

        I.append(sum(integrals))
        if k > 1:
            logger.debug(
                "Iteration=%i, integral=%f, err=%f"
                % (
                    k,
                    I[-1],
                    I[-2] - I[-1],
                )
            )
        else:
            logger.debug("Iteration=%i, integral=%f" % (k, I[-1]))

        if callback is not None:
            err = abs(I[-2] - I[-1]) if k > 1 else None
            if callback(I[-1], err, k):
                break

    return I[-1], abs(I[-2] - I[-1])


def _quad_prod(
    C,  # noqa: N803
    f,
    df=None,
    phi=None,
    psi=None,
    abs_tol=1.49e-08,
    rel_tol=1.49e-08,
    df_approx_order=2,
):
    if df is None:
        df = numdifftools.Derivative(f, order=df_approx_order)
        # df = lambda z: scipy.misc.derivative(f, z, dx=1e-8, n=1, order=3)

        # Too slow
        # import numdifftools.fornberg as ndf
        # ndf.derivative returns an array [f, f', f'', ...]
        # df = np.vectorize(lambda z: ndf.derivative(f, z, n=1)[1])

    integral, err = 0, 0
    for segment in C.segments:
        integrand_cache = {}

        def integrand(t):
            if t in integrand_cache.keys():
                i = integrand_cache[t]
            else:
                z = segment(t)
                i = (df(z) / f(z)) / (2j * pi) * segment.dzdt(t)
                if phi is not None:
                    i = phi(z) * i
                if psi is not None:
                    i = psi(z) * i
                integrand_cache[t] = i
            return i

        segment_integral, segment_err = integrate_quad_complex(
            integrand, 0, 1, epsabs=abs_tol, epsrel=rel_tol
        )
        integral += segment_integral
        err += segment_err

    return integral, abs(err)


class RootError(RuntimeError):
    pass


def count_roots(
    C,  # noqa: N803
    f,
    df=None,
    int_abs_tol=0.07,
    integer_tol=0.1,
    div_min=3,
    div_max=15,
    df_approx_order=2,
    int_method="quad",
):
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
    C : :class:`Contour <cxroots.Contour.Contour>`
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
        Only used if int_method='romb'.  The maximum number of divisions
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
    logger.info("Computing number of roots within " + str(C))

    with warnings.catch_warnings():
        # ignore warnings and catch if integral is NaN later
        warnings.simplefilter("ignore")
        integral, err = prod(
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

    if int_method == "romb":
        C._num_divisions_for_N = int(np.log2(len(C.segments[0]._trap_cache[f]) - 1))

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
        return num_zeros

    else:
        raise RootError("The number of enclosed roots has not converged to an integer")
