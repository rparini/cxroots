import logging
from typing import Callable, Optional, Tuple, Union

from mpmath import mp, mpmathify
from mpmath.calculus.optimization import Muller
from numpy import inf

Callback = Callable[[complex, complex, complex, int], bool]
ScalarCxFunc = Callable[[Union[complex, float]], complex]


def iterate_to_root(
    x0: complex,
    f: ScalarCxFunc,
    df: Optional[ScalarCxFunc] = None,
    step_tol: float = 1e-12,
    root_tol: float = 1e-12,
    max_iter: int = 20,
    refine_roots_beyond_tol: bool = False,
    callback: Optional[Callback] = None,
) -> Optional[complex]:
    """
    Starting with initial point x0 iterate to a root of f. This function is called
    during the rootfinding process to refine any roots found. If df is given then
    the Newton-Raphson method, :func:`~cxroots.iterative_methods.newton`, will be used,
    otherwise Muller's method, :func:`~cxroots.iterative_methods.muller`, will be used
    instead.

    Parameters
    ----------
    x0 : complex
        An initial point for the iteration.
    f : function
        Function of a single variable which we seek to find a root of.
    df : function, optional
        The derivative of f.
    step_tol: float, optional
        The routine ends if the step size, dx, between sucessive
        iterations satisfies abs(dx) < step_tol and refine_roots_beyond_tol is False.
    root_tol: float, optional
        A root, x, is only returned if abs(f(x)) < root_tol, otherwise None is returned
    max_iter : int, optional
        The routine ends after max_iter iterations.
    refine_roots_beyond_tol : bool, optional
        If True then the routine ends only once the error of the previous iteration,
        x0, was at least as good as the current iteration, x, in the sense that
        abs(f(x)) >= abs(f(x0)), and the previous iteration satisfied
        abs(dx0) < step_tol. In this case the previous iteration is returned as the
        approximation of the root, provided that it satisfies abs(f(x)) < root_tol
    callback : function, optional
        After each iteration callback(x, dx, f(x), iteration) will be
        called where 'x' is the current iteration of the estimated root,
        'dx' is the step size between the previous and current 'x' and
        'iteration' the number of iterations that have been taken.  If
        the callback function evaluates to True then the routine will end.

    Returns
    -------
    complex
        An approximation for a root of f. If the rootfinding was
        unsucessful then None will be returned instead.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Refining root: " + str(x0))

    if df is not None:
        try:
            root, err = newton(
                x0, f, df, step_tol, 0, max_iter, refine_roots_beyond_tol, callback
            )
        except (RuntimeError, OverflowError):
            return None
    else:
        # Muller's method:
        x1, x2, x3 = x0, x0 * (1 + 1e-8) + 1e-8j, x0 * (1 - 1e-8) - 1e-8j
        root, err = muller(
            x1,
            x2,
            x3,
            f,
            step_tol,
            0,
            max_iter,
            refine_roots_beyond_tol,
            callback,
        )

    if err < root_tol:
        return root


def muller(
    x1: complex,
    x2: complex,
    x3: complex,
    f: ScalarCxFunc,
    step_tol: float = 1e-12,
    root_tol: float = 0,
    max_iter: int = 20,
    refine_roots_beyond_tol: bool = False,
    callback: Optional[Callback] = None,
) -> Tuple[complex, float]:
    """
    A wrapper for mpmath's implementation of Muller's method.

    Parameters
    ----------
    x1 : float or complex
        An initial point for iteration, should be close to a root of f.
    x2 : float or complex
        An initial point for iteration, should be close to a root of f.
        Should not equal x1.
    x3 : float or complex
        An initial point for iteration, should be close to a root of f.
        Should not equal x1 or x2.
    f : function
        Function of a single variable which we seek to find a root of.
    step_tol: float, optional
        The routine ends if the step size, dx, between sucessive
        iterations satisfies abs(dx) < step_tol and refine_roots_beyond_tol is False.
    root_tol: float, optional
        The routine ends if abs(f(x)) < root_tol and refine_roots_beyond_tol is False.
    max_iter : int, optional
        The routine ends after max_iter iterations.
    refine_roots_beyond_tol : bool, optional
        If True then routine ends if the error of the previous iteration,
        x0, was at least as good as the current iteration, x, in the
        sense that abs(f(x)) >= abs(f(x0)) and the previous iteration
        satisfied either abs(dx0) < step_tol or abs(f(x0)) < root_tol.  In
        this case the previous iteration is returned as the approximation
        of the root.
    callback : function, optional
        After each iteration callback(x, dx, f(x), iteration) will be
        called where 'x' is the current iteration of the estimated root,
        'dx' is the step size between the previous and current 'x' and
        'iteration' the number of iterations that have been taken.  If
        the callback function evaluates to True then the routine will end.

    Returns
    -------
    complex
        The approximation to a root of f.
    float
        abs(f(x)) where x is the final approximation for the root of f.
    """
    logger = logging.getLogger(__name__)

    # mpmath insists on functions accepting mpc
    def f_mpmath(z):
        return mpmathify(f(complex(z)))

    mull = Muller(mp, f_mpmath, (x1, x2, x3), verbose=False)
    x0 = x3

    x, err = x0, abs(f(x0))
    err0, dx0 = inf, inf
    try:
        for iteration, (x, dx) in enumerate(mull):
            y = f_mpmath(x)
            err = abs(y)
            logger.debug(
                str(iteration)
                + " x="
                + str(x)
                + " |f(x)|="
                + str(err)
                + " dx="
                + str(dx)
            )

            if callback is not None and callback(x, dx, y, iteration + 1):
                break

            if (
                not refine_roots_beyond_tol
                and (abs(dx) < step_tol or err < root_tol)
                or iteration > max_iter
            ):
                break

            if (
                refine_roots_beyond_tol
                and (abs(dx0) < step_tol or err0 < root_tol)
                and err >= err0
            ):
                # The previous iteration was a better appproximation the current one so
                # assume that that was as close to the root as we are going to get.
                x, err = x0, err0
                break

            x0 = x

            if refine_roots_beyond_tol:
                # record previous error for comparison
                dx0, err0 = dx, err

    except ZeroDivisionError:
        # ZeroDivisionError comes up if the error is evaluated to be zero
        pass

    # cast mpc and mpf back to regular complex and float
    logger.debug(
        "Final approximation: x=" + str(complex(x)) + " |f(x)|=" + str(float(err))
    )
    return complex(x), float(err)


def newton(
    x0: complex,
    f: ScalarCxFunc,
    df: ScalarCxFunc,
    step_tol: float = 1e-12,
    root_tol: float = 0,
    max_iter: int = 20,
    refine_roots_beyond_tol: bool = False,
    callback: Optional[Callback] = None,
) -> Tuple[complex, float]:
    """
    Find an approximation to a point xf such that f(xf)=0 for a
    scalar function f using Newton-Raphson iteration starting at
    the point x0.

    Parameters
    ----------
    x0 : float or complex
        Initial point for Newton iteration, should be as close as
        possible to a root of f
    f : function
        Function of a single variable which we seek to find a root of.
    df : function
        Function of a single variable, df(x), providing the
        derivative of the function f(x) at the point x
    step_tol: float, optional
        The routine ends if the step size, dx, between sucessive
        iterations satisfies abs(dx) < step_tol and refine_roots_beyond_tol is False.
    root_tol: float, optional
        The routine ends if abs(f(x)) < root_tol and refine_roots_beyond_tol is False.
    max_iter : int, optional
        The routine ends after max_iter iterations.
    refine_roots_beyond_tol : bool, optional
        If True then routine ends if the error of the previous iteration,
        x0, was at least as good as the current iteration, x, in the
        sense that abs(f(x)) >= abs(f(x0)) and the previous iteration
        satisfied either abs(dx0) < step_tol or abs(f(x0)) < root_tol.  In
        this case the previous iteration is returned as the approximation
        of the root.
    callback : function, optional
        After each iteration callback(x, dx, f(x), iteration) will be
        called where 'x' is the current iteration of the estimated root,
        'dx' is the step size between the previous and current 'x' and
        'iteration' the number of iterations that have been taken.  If
        the callback function evaluates to True then the routine will end.

    Returns
    -------
    complex
        The approximation to a root of f.
    float
        abs(f(x)) where x is the final approximation for the root of f.
    """
    logger = logging.getLogger(__name__)
    x, y = x0, f(x0)
    dx0, y0 = inf, y

    for iteration in range(max_iter):
        dx = -y / df(x)
        x += dx
        y = f(x)

        logger.debug("x=" + str(x) + " f(x)=" + str(y) + " dx=" + str(dx))

        if callback is not None and callback(x, dx, y, iteration + 1):
            break

        if not refine_roots_beyond_tol and (abs(dx) < step_tol or abs(y) < root_tol):
            break

        if (
            refine_roots_beyond_tol
            and (abs(dx0) < step_tol or abs(y0) < root_tol)
            and abs(y) >= abs(y0)
        ):
            break

        if refine_roots_beyond_tol:
            # store previous dx and y
            dx0, y0 = dx, y

    logger.debug("Final approximation: x=" + str(x) + " |f(x)|=" + str(abs(y)))
    return x, abs(y)


def secant(
    x1: complex,
    x2: complex,
    f: ScalarCxFunc,
    step_tol: float = 1e-12,
    root_tol: float = 0,
    max_iter: int = 30,
    callback: Optional[Callback] = None,
) -> Tuple[complex, float]:
    """
    Find an approximation to a point xf such that f(xf)=0 for a
    scalar function f using the secant method.  The method requires
    two initial points x1 and x2, ideally close to a root,
    and proceeds iteratively.

    Parameters
    ----------
    x1 : float or complex
        An initial point for iteration, should be close to a
        root of f.
    x2 : float or complex
        An initial point for iteration, should be close to a
        root of f.  Should not equal x1.
    f : function
        Function of a single variable which we seek to find a root of.
    step_tol: float, optional
        The routine ends if the step size, dx, between sucessive
        iterations satisfies abs(dx) < step_tol and refine_roots_beyond_tol is False.
    root_tol: float, optional
        The routine ends if abs(f(x)) < root_tol and refine_roots_beyond_tol is False.
    max_iter : int, optional
        The routine ends after max_iter iterations.
    callback : function, optional
        After each iteration callback(x, dx, f(x), iteration) will be
        called where 'x' is the current iteration of the estimated root,
        'dx' is the step size between the previous and current 'x' and
        'iteration' the number of iterations that have been taken.  If
        the callback function evaluates to True then the routine will end.

    Returns
    -------
    complex
        The approximation to a root of f.
    float
        abs(f(x)) where x is the final approximation for the root of f.
    """
    # As in "Numerical Recipies 3rd Edition" pick the bound with the
    # smallest function value as the most recent guess
    y1, y2 = f(x1), f(x2)
    if abs(y1) < abs(y2):
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    for iteration in range(max_iter):
        dx = -(x2 - x1) * y2 / (y2 - y1)
        x1, x2 = x2, x2 + dx
        y1, y2 = y2, f(x2)

        if callback is not None and callback(x2, dx, y2, iteration + 1):
            break

        if abs(dx) < step_tol or abs(y2) < root_tol:
            break

    return x2, abs(y2)
