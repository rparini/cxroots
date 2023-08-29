import functools
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, NamedTuple, Optional

import numpy as np
from numpydoc.docscrape import FunctionDoc
from rich.progress import BarColumn, Progress, TextColumn

from .contour_interface import ContourABC
from .iterative_methods import iterate_to_root
from .root_approximation import approximate_roots
from .root_counting import RootError
from .root_result import RootResult
from .types import AnalyticFunc, IntegrationMethod
from .util import NumberOfRootsChanged, update_docstring


class MultiplicityError(RuntimeError):
    pass


class CountCalls:
    """
    Count how many times a given function is called.
    """

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func
        self.calls = 0
        self.points = 0

    def __call__(self, z: Any) -> Any:
        self.calls += 1
        if hasattr(z, "__len__"):
            self.points += len(z)
        else:
            self.points += 1
        return self.func(z)


@dataclass
class ContourData:
    contour: ContourABC
    num_roots: int


def make_contour_data(
    contours: List[ContourABC], num_roots: Dict[ContourABC, int]
) -> List[ContourData]:
    return [ContourData(contour=c, num_roots=num_roots[c]) for c in contours]


class RootFinderState(NamedTuple):
    roots: List[complex]
    multiplicities: List[int]
    contour_data: List[ContourData]
    num_remaining_roots: int


def find_roots_gen(
    original_contour: ContourABC,
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    guess_roots: Iterable[complex] = [],
    guess_roots_symmetry: Optional[Callable[[complex], Iterable[complex]]] = None,
    newton_step_tol: float = 1e-14,
    refine_roots_beyond_tol: bool = True,
    newton_max_iter: int = 50,
    root_err_tol: float = 1e-10,
    abs_tol: float = 1.49e-08,
    rel_tol: float = 1.49e-08,
    integer_tol: float = 0.1,
    int_abs_tol: float = 0.07,
    M: int = 5,  # noqa: N803
    err_stop: float = 1e-10,
    int_method: IntegrationMethod = "quad",
    div_min: int = 3,
    div_max: int = 15,
    df_approx_order: int = 2,
) -> Generator[RootFinderState, None, None]:
    """
    A generator which at each step takes a contour and either finds all
    the zeros of f within it or subdivides it further.  Based on the
    algorithm in [KB]_.

    Parameters
    ----------
    original_contour : :class:`Contour <cxroots.contour.Contour>`
        The contour which bounds the region in which all the roots of
        f(z) are sought.
    f : function
        A function of a single complex variable, z, which is analytic
        within the contour and has no poles or roots on the contour.
    df : function, optional
        A function of a single complex variable which is the derivative
        of the function f(z). If df is not given then it will be
        approximated with a finite difference formula.
    guess_roots : list, optional
        A list of known roots or guesses for roots (they are checked
        before being accepted).
    guess_roots_symmetry : function, optional
        A function of a single complex variable, z, which returns a list
        of all points which are expected to be roots of f, given that z
        is a root of f.
    newton_step_tol : float, optional
        The required accuracy of the root.  The iterative method used to
        give a final value for each root will exit if the step size, dx,
        between sucessive iterations satisfies abs(dx) < newton_step_tol
        and refine_roots_beyond_tol is False.
    refine_roots_beyond_tol : bool, optional
        If True then the iterative method used to refine the roots will
        exit when error of the previous iteration, x0, was at least as
        good as the current iteration, x, in the sense that
        abs(f(x)) >= abs(f(x0)) and the previous iteration satisfied
        abs(dx0) < newton_step_tol.  In this case the preivous iteration
        is returned as the approximation of the root.
    newton_max_iter : int, optional
        The iterative method used to give a final value for each root
        will exit if the number of iterations exceeds newton_max_iter.
    root_err_tol : float, optional
        A complex value z is considered a root if abs(f(z)) < root_err_tol
    abs_tol : float, optional
        Absolute error tolerance used by the contour integration.
    rel_tol : float, optional
        Relative error tolerance used by the contour integration.
    integer_tol : float, optional
        A number is considered an integer if it is within integer_tol of
        an integer.  Used when determing if the value for the number of
        roots within a contour and the values of the computed
        multiplicities of roots are acceptably close to integers.
    int_abs_tol : float, optional
        The absolute error tolerance used for the contour integration
        when determining the number of roots within a contour.  Since
        the result of this integration must be an integer it can be much
        less accurate than usual.
    M : int, optional
        If the number of roots (including multiplicites) within a contour
        is greater than M then the contour is subdivided further.  M must
        be greater than or equal to the largest multiplcity of any root.
    err_stop : float, optional
        The number of distinct roots within a contour, n, is determined
        by checking if all the elements of a list of contour integrals
        involving formal orthogonal polynomials are sufficently close to
        zero, ie. that the absolute value of each element is < err_stop.
        If err_stop is too large/small then n may be smaller/larger than
        it actually is.
    int_method : {'quad', 'romb'}, optional
        If 'quad' then :func:`scipy.integrate.quad` is used to perform the
        integral.  If 'romb' then Romberg integraion, using
        :func:`scipy.integrate.romb`, is performed instead.  Typically, quad is
        the better choice but it requires that the real and imaginary
        parts of each integral are calculated sepeartely, in addition,
        if df is not provided, 'quad' will require additional function
        evaluations to approximate df at each point that f is evaluated
        at.  If evaluating f is expensive then 'romb' may be more
        efficient since it computes the real and imaginary parts
        simultaniously and if df is not provided it will approximate it
        using only the values of f that would be required by the
        integration routine in any case.
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
        passed to :func:`numdifftools.Derivative` and is the order of the error
        term in the Taylor approximation.  df_approx_order must be even.

    Yields
    ------
    list[complex]
        Roots of f(z) within the contour original_contour
    list[int]
        Multiplicites of roots
    list[(contour, roots_in_contour)]
        The contours which still contain roots
    int
        Remaining number of roots to be found within the contour

    References
    ----------
    .. [KB] Peter Kravanja, Marc Van Barel, "Computing the Zeros of
        Anayltic Functions", Springer (2000)
    """
    from .contours.circle import Circle

    # wrap f to record the number of function calls
    f = CountCalls(f)

    count_kwargs = {
        "f": f,
        "df": df,
        "int_abs_tol": int_abs_tol,
        "integer_tol": integer_tol,
        "div_min": div_min,
        "div_max": div_max,
        "df_approx_order": df_approx_order,
        "int_method": int_method,
    }

    logger = logging.getLogger(__name__)
    roots: List[complex] = []
    multiplicities: List[int] = []
    failed_contours: List[ContourABC] = []
    contours: List[ContourABC] = []
    num_roots: Dict[ContourABC, int] = {}

    try:
        # compute the total number of zeros, including multiplicities, within the
        # originally given contour
        num_roots[original_contour] = original_contour.count_roots(**count_kwargs)
        contours.append(original_contour)
    except RuntimeError as e:
        raise RuntimeError(
            """
            Integration along the initial contour has failed. There is likely a root on
            or close to the initial contour. Try changing the initial contour, if
            possible. Changing the integration method (int_method) may also help.
            """
        ) from e

    def subdivide(parent: ContourABC) -> None:
        """Given a contour, parent_contour, subdivide it into multiple contours."""
        logger.info("Subdividing " + str(parent))

        for subcontours in parent.subdivisions():
            # if a contour has already been used and caused an error then skip it
            failed_contour_desc = list(map(str, failed_contours))
            if any(
                contour_desc in failed_contour_desc
                for contour_desc in list(map(str, subcontours))
            ):
                continue

            # if a contour is near to a known root then skip this subdivision
            for root in roots:
                if any(contour.distance(root) < 0.01 for contour in subcontours):
                    continue

            try:
                subcontour_num_roots = [
                    contour.count_roots(**count_kwargs)
                    for contour in np.array(subcontours)
                ]
                while (
                    num_roots[parent] != sum(subcontour_num_roots)
                    and int_abs_tol > 0.01
                ):
                    logger.warning(
                        "Number of roots in sub contours not adding up to parent "
                        "contour. Recomputing number of roots in parent and child "
                        f"contours with int_abs_tol={0.5 * int_abs_tol}"
                    )
                    temp_count_kwargs = count_kwargs.copy()
                    temp_count_kwargs["int_abs_tol"] *= 0.5
                    num_roots[parent] = parent.count_roots(**temp_count_kwargs)
                    subcontour_num_roots = [
                        contour.count_roots(**temp_count_kwargs)
                        for contour in np.array(subcontours)
                    ]

                if num_roots[parent] == sum(subcontour_num_roots):
                    break

            except RootError:
                # If the number of zeros within either of the new contours is not an
                # integer then it is likely that the introduced line which subdivides
                # 'parent_contour' lies on a zero. To avoid this we will try to place
                # the subdividing line at a different point along the division axis
                logger.warning(
                    "RootError encountered when subdivding "
                    + str(parent)
                    + " into:\n"
                    + str(subcontours[0])
                    + "\n"
                    + str(subcontours[1])
                )
                continue
        else:
            # The list of possible subdivisions has been exhaused
            raise RuntimeError(
                f"""Unable to subdivide contour:
                \t{parent}
                """
            )

        # add sub-contours with roots to the list of contours to find the roots in
        for n_roots, c in zip(subcontour_num_roots, subcontours):
            if n_roots != 0:
                contours.append(c)
                num_roots[c] = n_roots

    def reverse_subdivision(contour: ContourABC) -> None:
        """
        Remove the contour and all its siblings and children from the list of contours
        to be examined/subdivided and put its parent back in the list to be subdivided
        """
        if contour.parent is None:
            raise ValueError("Contour has no parent so subdivision cannot be reversed")

        relations = []
        if contour.parent.children is not None:
            # The contour has siblings
            relations.extend([c for c in contour.parent.children if c != contour])

        if contour.children is not None:
            relations.extend(contour.children)

        for c in contours.copy():
            if c in relations:
                contours.remove(c)

        # put the parent contour back into the list of contours to subdivide again
        contours.append(contour.parent)

        # do not use this contour again
        failed_contours.append(contour)

    def add_root(root: complex, multiplicity: int) -> None:
        # check that the root we have found is distinct from the ones we already have
        if not roots or np.all(abs(np.array(roots) - root) > newton_step_tol):
            # add the root to the list if it is within the original contour
            if original_contour.contains(root):
                roots.append(root)
                multiplicities.append(multiplicity)
                logger.info(
                    "Recorded root "
                    + str(root)
                    + " with multiplicity "
                    + str(multiplicity)
                )
            else:
                logger.debug(
                    "Root " + str(root) + " ignored as not within original contour"
                )

            # check to see if there are any other roots implied by the given symmetry
            if guess_roots_symmetry is not None:
                for x0 in guess_roots_symmetry(root):
                    # first check that x0 is distinct from the roots we already have
                    if np.all(abs(np.array(roots) - x0) > newton_step_tol):
                        logger.info(str(root) + " is a root so checking " + str(x0))
                        new_root = iterate_to_root(
                            x0,
                            f,
                            df,
                            newton_step_tol,
                            root_err_tol,
                            newton_max_iter,
                            refine_roots_beyond_tol,
                        )
                        if new_root is not None:
                            circle = Circle(new_root, 1e-3)
                            contours.append(circle)
                            num_roots[circle] = circle.count_roots(**count_kwargs)
        else:
            logger.debug("Already recorded root " + str(root))

    # Add contours surrounding known roots so that they will be checked
    for root in guess_roots:
        circle = Circle(root, 1e-3)
        num_roots[circle] = circle.count_roots(**count_kwargs)
        contours.append(circle)

    while contours:
        # yield the initial state here so that the animation in demo_find_roots shows
        # the first frame
        num_found_roots = sum(multiplicities)
        remaining_roots = num_roots[original_contour] - num_found_roots
        yield RootFinderState(
            roots,
            multiplicities,
            make_contour_data(contours, num_roots),
            remaining_roots,
        )
        contour = contours.pop()

        # if a known root is too near to this contour then reverse the subdivision that
        # created it
        if np.any([contour.distance(root) < newton_step_tol for root in roots]):
            reverse_subdivision(contour)
            continue

        # if the contour is smaller than the newton_step_tol then just assume that
        # the root is at the center of the contour, print a warning and move on
        if contour.area < newton_step_tol:
            root = iterate_to_root(
                contour.central_point,
                f,
                df,
                newton_step_tol,
                root_err_tol,
                newton_max_iter,
                refine_roots_beyond_tol,
            )
            if (
                root is None
                or abs(f(root)) > abs(f(contour.central_point))
                or not contour.contains(root)
            ):
                root = contour.central_point

            warnings.warn(
                "The area of this contour is smaller than newton_step_tol. Try "
                f"increasing root_tol. The point z = {root.real} + {root.imag}i has "
                f"been recorded as a root of multiplicity {num_roots[contour]}. "
                f"The error |f(z)| = {abs(f(root))}"
            )
            add_root(root, num_roots[contour])
            continue

        # if all the roots within the contour have been located then continue to the
        # next contour
        num_known_roots = sum(
            multiplicity
            for root, multiplicity in zip(roots, multiplicities)
            if contour.contains(root)
        )
        if num_roots[contour] == num_known_roots:
            continue

        # if there are too many roots within the contour then subdivide or
        # if there are any known roots within the contour then also subdivide
        # (so as not to waste time re-approximating these roots)
        if num_known_roots > 0 or num_roots[contour] > M:
            subdivide(contour)
            continue

        # Approximate the roots in this contour
        if int_method == "romb":
            # Check to see if the number of roots has changed after new values of f
            # have been sampled
            new_count_kwargs = count_kwargs.copy()

            def callback(integral, err, num_div):
                if num_div > new_count_kwargs["div_min"]:
                    logger.info(
                        "Checking root count using the newly sampled values of f"
                    )
                    new_count_kwargs["div_min"] = num_div
                    new_num_roots = contour.count_roots(**new_count_kwargs)

                    if new_num_roots != num_roots[contour]:
                        logger.info(
                            "Number of roots has been recalculated using more samples"
                            " of f"
                        )
                        original_num_roots = num_roots[contour]
                        num_roots[contour] = new_num_roots
                        raise NumberOfRootsChanged(
                            "The additional function evaluations of f taken while "
                            "approximating the roots within the contour have "
                            "shown that the number of roots of f within the contour "
                            f"is {new_num_roots} instead of {original_num_roots}."
                        )

        else:

            def callback(integral, err, num_div):
                pass

        try:
            approx_roots, approx_multiplicities = approximate_roots(
                contour,
                num_roots[contour],
                f,
                df,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                err_stop=err_stop,
                div_min=div_min,
                div_max=div_max,
                df_approx_order=df_approx_order,
                root_tol=newton_step_tol,
                int_method=int_method,
                callback=callback,
            )
        except NumberOfRootsChanged:
            logger.debug("The number of roots within the contour has been reevaluated")
            if num_roots[contour] > M:
                subdivide(contour)
            else:
                contours.append(contour)
            continue

        for approx_root, approx_multiplicity in list(
            zip(approx_roots, approx_multiplicities)
        ):
            # check that the multiplicity is close to an integer
            multiplicity = round(approx_multiplicity.real)
            if (
                abs(multiplicity - approx_multiplicity.real) > integer_tol
                or abs(approx_multiplicity.imag) > integer_tol
                or multiplicity < 1
            ):
                continue

            # attempt to refine the root
            root = iterate_to_root(
                approx_root,
                f,
                df,
                newton_step_tol,
                root_err_tol,
                newton_max_iter,
                refine_roots_beyond_tol,
            )

            if root is None or abs(f(approx_root)) < abs(f(root)):
                # stick with the original approximation
                root = approx_root

            if abs(f(root)) < root_err_tol:
                if np.any(
                    np.abs(np.round(approx_multiplicities) - approx_multiplicities)
                    > integer_tol
                ):
                    # the computed multiplicity might be unreliable so make a contour
                    # focused on that point instead so that the multiplicity will be
                    # recomputed. If the current contour is already centered on that
                    # point then make one with a smaller radius
                    radius = (
                        min(contour.radius * 0.1, 1e-3)
                        if isinstance(contour, Circle) and contour.center == root
                        else 1e-3
                    )
                    circle = Circle(root, radius)
                    contours.append(circle)
                    num_roots[circle] = circle.count_roots(**count_kwargs)
                else:
                    add_root(root, multiplicity)

            # if the root turns out to be very close to the contour then this may have
            # introduced an error.  Therefore, compute the multiplicity of this root
            # directly and disregard this contour (repeat its parent's subdivision).
            if contour.distance(root) < newton_step_tol:
                # remove the contour and any relations
                reverse_subdivision(contour)

        # if we haven't found all the roots then subdivide further
        num_known_roots = sum(
            multiplicity
            for root, multiplicity in zip(roots, multiplicities)
            if contour.contains(root)
        )
        if num_roots[contour] != num_known_roots and contour not in failed_contours:
            subdivide(contour)

    result = RootResult(roots, multiplicities, original_contour)
    logger.info(
        "Completed rootfinding with "
        + str(f.calls)
        + " evaluations of the given analytic function at "
        + str(f.points)
        + " points\n"
        + str(result)
    )

    num_found_roots = sum(multiplicities)
    remaining_roots = num_roots[original_contour] - num_found_roots
    yield RootFinderState(
        roots, multiplicities, make_contour_data(contours, num_roots), remaining_roots
    )


@update_docstring(Parameters=FunctionDoc(find_roots_gen)["Parameters"])
@functools.wraps(find_roots_gen, assigned=("__module__", "__name__"))
def find_roots(
    original_contour: ContourABC,
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    verbose: bool = False,
    **kwargs,
) -> RootResult:
    """
    Find all the roots of the complex analytic function f within the
    given contour.

    Parameters
    ----------
    %(find_roots_gen.parameters)s
    verbose : bool, optional
        If True print a progress bar showing the rootfinding progress.

    Returns
    -------
    result : :class:`RootResult <cxroots.root_result.RootResult>`
        A container for the roots and their multiplicities.
    """
    roots, multiplicities = [], []
    root_finder = find_roots_gen(original_contour, f, df, **kwargs)

    if verbose:
        text_column = TextColumn("{task.description}")
        bar_column = BarColumn(bar_width=None)
        progress_text_column = TextColumn(
            "{task.completed} of {task.total} roots found"
        )
        progress = Progress(text_column, bar_column, progress_text_column, expand=True)
        task = progress.add_task("Rootfinding")
        progress.start()

        try:
            for roots, multiplicities, _, num_remaining_roots in root_finder:
                num_found_roots = sum(multiplicities)
                total_roots = num_found_roots + num_remaining_roots
                progress.update(task, completed=num_found_roots, total=total_roots)
        finally:
            progress.stop()

    else:
        for roots, multiplicities, _, _ in root_finder:
            pass

    return RootResult(roots, multiplicities, original_contour)
