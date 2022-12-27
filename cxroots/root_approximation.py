import functools
import logging
from typing import Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import scipy.linalg

from .contour_interface import ContourABC
from .root_counting import RombCallback, prod
from .types import AnalyticFunc, ComplexScalarOrArray, IntegrationMethod, ScalarOrArray


def approximate_roots(
    C: ContourABC,  # noqa: N803
    N: int,  # noqa: N803
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    err_stop: float = 1e-10,
    div_min: int = 3,
    div_max: int = 15,
    df_approx_order: int = 2,
    root_tol: float = 1e-8,
    int_method: IntegrationMethod = "quad",
    callback: Optional[RombCallback] = None,
) -> Tuple[Tuple[complex, ...], Tuple[float, ...]]:
    """
    Approximate the roots and multiplcities of the function f within the
    contour C using the method of [KB]_.  The multiplicites are computed
    using eq. (21) in [SLV]_.

    Parameters
    ----------
    C : :class:`~<cxroots.contour.Contour>`
        The contour which encloses the roots of f the user wishes to find.
    N : int
        The number of roots (counting multiplicties) of f within C.
        This is the result of calling :meth:`~cxroots.contour.Contour.count_roots`.
    f : function
        The function for which the roots are sought.  Must be a function
        of a single complex variable, z, which is analytic within C and
        has no poles or roots on the C.
    df : function, optional
        A function of a single complex variable which is the derivative
        of the function f(z). If df is not given then it will be
        approximated with a finite difference formula.
    abs_tol : float, optional
        Absolute error tolerance for integration.
    rel_tol : float, optional
        Relative error tolerance for integration.
    err_stop : float, optional
        The number of distinct roots within a contour, n, is determined
        by checking if all the elements of a list of contour integrals
        involving formal orthogonal polynomials are sufficently close to
        zero, ie. that the absolute value of each element is < err_stop.
        If err_stop is too large/small then n may be smaller/larger than
        it actually is.
    div_min : int, optional
        If the Romberg integration method is used then div_min is the
        minimum number of divisions before the Romberg integration
        routine is allowed to exit.
    div_max : int, optional
        If the Romberg integration method is used then div_max is the
        maximum number of divisions before the Romberg integration
        routine exits.
    df_approx_order : int, optional
        Only used if df=None and method='quad'.  The argument order=df_approx_order is
        passed to numdifftools.Derivative and is the order of the error
        term in the Taylor approximation.  df_approx_order must be even.
    root_tol : float, optional
        If any roots are within root_tol of one another then they will be
        treated as duplicates and removed.  This helps to alleviate the
        problem of err_stop being too small.
    int_method : {'quad', 'romb'}, optional
        If 'quad' then :func:`scipy.integrate.quad` is used to perform
        integration.  If 'romb' then Romberg integraion is performed
        instead.
    callback : function, optional
        Only used if int_method is 'romb'. Passed to
        :func:`~<cxroots.root_counting.prod>`.

    Returns
    -------
    tuple of complex
        The distinct roots of f within the contour C.
    tuple of float
        The corresponding multiplicites of the roots within C.  Should
        be integers but will not be automatically rounded here.

    References
    ----------
    .. [KB] P. Kravanja and M. Van Barel. "Computing the Zeros of
        Anayltic Functions". Springer (2000)
    .. [SLV] E. Strakova, D. Lukas, P. Vodstrcil. "Finding Zeros of
        Analytic Functions and Local Eigenvalue Analysis Using Contour
        Integral Method in Examples". Mathematical Analysis and Numerical
        Mathematics, Vol. 15, 2, (2017)
    """
    logger = logging.getLogger(__name__)
    logger.info("Approximating the " + str(N) + " roots in: " + str(C))

    if N == 0:
        return (), ()

    product = functools.partial(
        prod,
        C,
        f,
        df,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        div_min=div_min,
        div_max=div_max,
        df_approx_order=df_approx_order,
        int_method=int_method,
        callback=callback,
    )

    s = [N, product(lambda z: z)]  # ordinary moments
    mu = s[1] / N
    phi_zeros = [np.array([]), np.array([mu])]

    def phi(i: int) -> AnalyticFunc:
        if len(phi_zeros[i]) == 0:
            return lambda z: complex(1)
        coeff = np.poly(phi_zeros[i])
        # We should be using Polynomial.fromroots but this is not hashable so
        # causes problems with caching
        return lambda z: np.polyval(coeff, z)
        # return npp.Polynomial.fromroots(phi_zeros[i])

    def phi1phi(i: int) -> AnalyticFunc:
        @overload
        def func(z: Union[complex, float]) -> complex:
            ...

        @overload
        def func(
            z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
        ) -> Union[npt.NDArray[np.complex_], complex]:
            ...

        def func(z: ScalarOrArray) -> ComplexScalarOrArray:
            return phi(1)(z) * phi(i)(z)

        return func

    # initialize G_{pq} = <phi_p, phi_q>
    G = np.zeros((N, N), dtype=np.complex128)  # noqa: N806
    G[0, 0] = N  # = <phi_0, phi_0> = <1,1>

    # initialize G1_{pq} = <phi_p, phi_1 phi_q>
    G1 = np.zeros((N, N), dtype=np.complex128)  # noqa: N806
    G1[0, 0] = 0  # = <phi_0, phi_1 phi_0> = <1, z-mu> = s1-mu*N = 0

    r, t = 1, 0
    while r + t < N:
        ### define FOP of degree r+t+1
        p = r + t
        G[p, 0 : p + 1] = [product(phi(p), phi(q)) for q in range(r + t + 1)]
        G[0 : p + 1, p] = G[p, 0 : p + 1]  # G is symmetric
        logger.debug("G=\n" + str(G[: p + 1, : p + 1]))

        G1[p, 0 : p + 1] = [product(phi(p), phi1phi(q)) for q in range(r + t + 1)]
        G1[0 : p + 1, p] = G1[p, 0 : p + 1]  # G1 is symmetric
        logger.debug("G1=\n" + str(G1[: p + 1, : p + 1]))

        """
        If any of the zeros of the FOP are outside of the interior
        of the contour then we assume that they are 'arbitary' and
        instead define the FOP as an inner polynomial. [KB]
        """
        poly_roots = scipy.linalg.eig(G1[: p + 1, : p + 1], G[: p + 1, : p + 1])[0] + mu
        if np.all([C.contains(z) for z in poly_roots]):
            r, t = r + t + 1, 0
            phi_zeros.append(poly_roots)
            logger.debug(
                "Regular polynomial " + str(r + t) + " roots: " + str(phi_zeros[-1])
            )

            # is the number of distinct roots, n=r?
            phi_func_last = phi(-1)
            for j in range(N - r):
                ip = product(lambda z: phi_func_last(z) * (z - mu) ** j, phi_func_last)
                logger.debug("%i of %i, abs(ip)=%f" % (j, N - r, abs(ip)))
                if abs(ip) > err_stop:
                    # n != r so carry on
                    logger.debug("n != " + str(r))
                    break
            else:
                # the for loop did not break
                logger.debug("n = " + str(r))
                break

        else:
            # define an inner polynomial as phi_{r+t+1} = phi_{t+1} phi_{r}
            t += 1
            phi_zeros.append(np.append(phi_zeros[t], phi_zeros[r]))
            logger.debug(
                "Inner polynomial " + str(r + t) + " roots: " + str(phi_zeros[-1])
            )

    roots = phi_zeros[-1]

    # remove any roots which are not distinct
    roots_to_remove = []
    for i, root in enumerate(roots):
        if len(roots[i + 1 :]) > 0 and np.any(np.abs(root - roots[i + 1 :]) < root_tol):
            roots_to_remove.append(i)
    roots = np.delete(roots, roots_to_remove)
    n = len(roots)

    ### compute the multiplicities, eq. (1.19) in [KB]
    # V = np.column_stack([roots**i for i in range(n)])
    # if n > 2: logger.debug('Computing ordinary moments')
    # s += [product(lambda z: z**p)[0] for p in range(2, n)]
    # multiplicities = np.dot(s[:n], np.linalg.inv(V))

    ### compute the multiplicities, eq. (21) in [SLV]
    V = np.array([[phi(j)(root) for root in roots] for j in range(n)])  # noqa: N806
    multiplicities = np.dot(np.linalg.inv(V), G[:n, 0])

    ### The method used in the vandermonde module doesn't seem significantly
    ### better than np.dot(s, np.linalg.inv(V)).  Especially since we know
    ### the result must be an integer anyway.
    # import vandermonde
    # multiplicities_vandermonde = vandermonde.solve_transpose(
    #     np.array(roots), np.array(s)
    # )

    ### Note that n = rank(H_N) is not used since calculating the
    ### rank of a matrix of floats can be quite unstable
    # s_func = lambda p: prod(C, f, df, lambda z: z**p)[0]
    # HN = np.fromfunction(np.vectorize(lambda p,q: s_func(p+q)), shape=(N,N))
    # print('n?', np.linalg.matrix_rank(HN, tol=1e-10))

    logger.debug(
        "Approximate (roots, multiplicities): " + str(zip(roots, multiplicities))
    )
    return tuple(roots), tuple(multiplicities)
