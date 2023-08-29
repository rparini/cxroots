import functools
from typing import Generator, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

from .contour_interface import ContourABC
from .paths import ComplexPath, ComplexPathType
from .root_counting import count_roots
from .root_finding import find_roots
from .root_finding_demo import demo_find_roots, demo_roots_animation
from .root_result import RootResult
from .types import AnalyticFunc
from .util import remove_para


class Contour(ContourABC):
    """
    A base class for contours in the complex plane.

    Attributes
    ----------
    central_point : complex
        The point at the center of the contour.
    area : float
        The surface area of the contour.
    """

    # Should be set in subclass
    axis_names = ()

    def __init__(self, segments: List[ComplexPathType]):
        self.segments = segments

        # A contour created by the subdvision method will have this attribute set to
        # the axis along which the line subdividing the parent contour was a constant.
        # This is done in order to implement the "alternating" subdivision method
        self._created_by_subdivision_axis: Optional[str] = None
        # _parent and _children are set in subdivision method
        self._parent: Optional[Contour] = None
        self._children: Optional[Sequence[Contour]] = None

    @overload
    def __call__(self, t: float) -> complex:
        ...

    @overload
    def __call__(self, t: npt.NDArray[np.float_]) -> npt.NDArray[np.complex_]:
        ...

    def __call__(
        self, t: Union[float, npt.NDArray[np.float_]]
    ) -> Union[complex, npt.NDArray[np.complex_]]:
        r"""
        The point on the contour corresponding the value of the
        parameter t.

        Parameters
        ----------
        t : float
            A real number :math:`0\leq t \leq 1` which parameterises
            the contour.

        Returns
        -------
        complex
            A point on the contour.

        Example
        -------

        >>> from cxroots import Circle
        >>> c = Circle(0,1) # Circle |z|=1 parameterised by e^{it}
        >>> c(0.25)
        (6.123233995736766e-17+1j)
        >>> c(0) == c(1)
        True
        """
        t = np.array(t, dtype=np.float_)
        num_segments = len(self.segments)
        segment_index = np.array(num_segments * t, dtype=int)
        segment_index = np.mod(segment_index, num_segments)

        if hasattr(segment_index, "__iter__"):
            return np.array(
                [
                    self.segments[i](num_segments * t[ti] % 1)
                    for ti, i in enumerate(segment_index)
                ],
                dtype=complex,
            )
        else:
            return self.segments[segment_index](num_segments * t % 1)

    def trap_product(self, *args, **integration_kwargs) -> complex:
        r"""
        Use Romberg integration to estimate the symmetric bilinear form used in
        (1.12) of [KB]_ using 2**k+1 samples

        .. math::

            <\phi,\psi> = \frac{1}{2\pi i} \oint_C \phi(z)\psi(z)\frac{f'(z)}{f(z)} dz
        """
        return sum(s.trap_product(*args, **integration_kwargs) for s in self.segments)

    functools.update_wrapper(
        trap_product, ComplexPath.trap_product, assigned=["__doc__", "__annotations__"]
    )

    def integrate(self, f: AnalyticFunc, **integration_kwargs) -> complex:
        r"""
        Integrate the function f along the contour C

        .. math::

            \oint_C f(z) dz
        """
        return sum(
            segment.integrate(f, **integration_kwargs) for segment in self.segments
        )

    functools.update_wrapper(
        integrate, ComplexPath.integrate, assigned=["__doc__", "__annotations__"]
    )

    def plot(self, *args, **kwargs) -> None:
        self.size_plot()
        for segment in self.segments:
            segment.plot(*args, **kwargs)

    functools.update_wrapper(
        plot, ComplexPath.plot, assigned=["__doc__", "__annotations__"]
    )

    def size_plot(self) -> None:
        """
        Adjust the plot axes limits to nicely frame the contour. Called as part of
        :meth:`~cxroots.contour.Contour.plot`
        """
        import matplotlib.pyplot as plt

        t = np.linspace(0, 1, 1000)
        z = self(t)
        xpad = (max(np.real(z)) - min(np.real(z))) * 0.1
        ypad = (max(np.imag(z)) - min(np.imag(z))) * 0.1

        xmin = min(np.real(z)) - xpad
        xmax = max(np.real(z)) + xpad
        ymin = min(np.imag(z)) - ypad
        ymax = max(np.imag(z)) + ypad
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    def show(self, save_file: Optional[str] = None, **plot_kwargs) -> None:
        """
        Shows the contour as a 2D plot in the complex plane.  Requires
        Matplotlib.

        Parameters
        ----------
        save_file : str (optional)
            If given then the plot will be saved to disk with name
            'save_file'.  If save_file=None the plot is shown on-screen.
        **plot_kwargs
            Key word arguments are as in :meth:`~cxroots.contour.Contour.plot`.
        """
        import matplotlib.pyplot as plt

        self.plot(**plot_kwargs)

        if save_file is not None:
            plt.savefig(save_file, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def subdivide(self, axis, division_factor: float) -> Tuple["Contour", ...]:
        """
        Subdivide the contour
        """
        raise NotImplementedError("subdivide must be implemented in a subclass")

    @property
    def parent(self) -> Optional["Contour"]:
        return self._parent

    @property
    def children(self) -> Optional[Sequence["Contour"]]:
        return self._children

    def subdivisions(
        self, axis: str = "alternating"
    ) -> Generator[Tuple["Contour", ...], None, None]:
        """
        A generator for possible subdivisions of the contour.

        Parameters
        ----------
        axis : str, 'alternating' or any element of self.axis_names.
            The axis along which the line subdividing the contour is a
            constant (eg. subdividing a circle along the radial axis
            will give an outer annulus and an inner circle).  If
            alternating then the dividing axis will always be different
            to the dividing axis used to create the contour which is now
            being divided.

        Yields
        ------
        tuple
            A tuple with two contours which subdivide the original
            contour.
        """
        if axis == "alternating":
            if self._created_by_subdivision_axis is None:
                axis_index = 0
            else:
                axis_index = (
                    self.axis_names.index(self._created_by_subdivision_axis) + 1
                ) % len(self.axis_names)

            axis = self.axis_names[axis_index]

        for division_factor in division_factor_gen():
            yield self.subdivide(axis, division_factor)

    def distance(self, z: complex) -> float:
        """
        The distance from the point z in the complex plane to the
        nearest point on the contour.

        Parameters
        ----------
        z : complex
            The point from which to measure the distance to the closest
            point on the contour to z.

        Returns
        -------
        float
            The distance from z to the point on the contour which is
            closest to z.
        """
        return min(segment.distance(z) for segment in self.segments)

    @remove_para("C")
    @functools.wraps(count_roots)
    def count_roots(
        self, f: AnalyticFunc, df: Optional[AnalyticFunc] = None, **kwargs
    ) -> int:
        return count_roots(self, f, df, **kwargs)

    @remove_para("original_contour")
    @functools.wraps(find_roots)
    def roots(
        self, f: AnalyticFunc, df: Optional[AnalyticFunc] = None, **kwargs
    ) -> RootResult:
        return find_roots(self, f, df, **kwargs)

    @remove_para("C")
    @functools.wraps(demo_find_roots)
    def demo_roots(self, *args, **kwargs) -> None:
        return demo_find_roots(self, *args, **kwargs)

    @remove_para("C")
    @functools.wraps(demo_roots_animation)
    def demo_roots_animation(self, *args, **kwargs):
        return demo_roots_animation(self, *args, **kwargs)


def division_factor_gen() -> Generator[float, None, None]:
    """A generator for division_factors."""
    yield 0.3  # being off-center is a better first choice for certain problems

    x = 0.5
    yield x
    for diff in np.linspace(0, 0.5, int(1 + 10 / 2.0))[1:-1]:
        yield x + diff
        yield x - diff
