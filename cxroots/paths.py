import functools
from math import pi
from typing import Optional, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import scipy.integrate

from .types import AnalyticFunc, Color, IntegrationMethod
from .util import integrate_quad_complex

ComplexPathType = TypeVar("ComplexPathType", bound="ComplexPath")


class ComplexPath(object):
    """A base class for paths in the complex plane."""

    def __init__(self):
        self._integral_cache = {}
        self._trap_cache = {}

        # If the path is created during subvision then this will be set to the path
        # that is oppositely oriented to this path. This is done so that we can look
        # up the _integral_cache for the reverse path
        self._reverse_path: Optional[ComplexPath] = None

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
        The parameterization of the path in the varaible :math:`t\in[0,1]`.

        Parameters
        ----------
        t : float
            A real number :math:`0\leq t \leq 1`.

        Returns
        -------
        complex
            A point on the path in the complex plane.
        """
        raise NotImplementedError("__call__ must be implemented in a subclass")

    @overload
    def dzdt(self, t: float) -> complex:
        ...

    @overload
    def dzdt(self, t: npt.NDArray[np.float_]) -> npt.NDArray[np.complex_]:
        ...

    def dzdt(
        self, t: Union[float, npt.NDArray[np.float_]]
    ) -> Union[complex, npt.NDArray[np.complex_]]:
        """
        The derivative of the parameterised curve in the complex plane, z, with
        respect to the parameterization parameter, t.
        """
        raise NotImplementedError("dzdt must be implemented in a subclass")

    def distance(self, z: complex) -> float:
        """
        Distance from the point z to the closest point on the line.
        """
        raise NotImplementedError("distance must be implemented in a subclass")

    def trap_values(
        self,
        f: AnalyticFunc,
        k: int,
        use_cache: bool = True,
    ) -> npt.NDArray[np.complex128]:
        """
        Compute or retrieve (if cached) the values of the functions f
        at :math:`2^k+1` points along the contour which are evenly
        spaced with respect to the parameterisation of the contour.

        Parameters
        ----------
        f : function
            A function of a single complex variable.
        k : int
            Defines the number of points along the curve that f is to be
            evaluated at as :math:`2^k+1`.
        use_cache : bool, optional
            If True then use, if available, the results of any previous
            calls to this function for the same f and save any new
            results so that they can be reused later.

        Returns
        -------
        :class:`numpy.ndarray`
            The values of f at :math:`2^k+1` points along the contour
            which are evenly spaced with respect to the parameterisation
            of the contour.
        """
        if f in self._trap_cache.keys() and use_cache:
            vals = self._trap_cache[f]
            vals_k = int(np.log2(len(vals) - 1))

            if vals_k == k:
                return vals
            elif vals_k > k:
                return vals[:: 2 ** (vals_k - k)]
            else:
                t = np.linspace(0, 1, 2**k + 1)
                vals = np.empty(2**k + 1, dtype=np.complex128)
                vals.fill(np.nan)
                vals[:: 2 ** (k - vals_k)] = self._trap_cache[f]
                vals[np.isnan(vals)] = f(self(t[np.isnan(vals)]))

                # cache values
                self._trap_cache[f] = vals
                return vals

        else:
            t = np.linspace(0, 1, 2**k + 1)
            vals = f(self(t))
            if not isinstance(vals, np.ndarray):
                # Handle case when f does not return a vector, perhaps
                # because the function is actually a constant
                vals = vals * np.ones(len(t), dtype=np.complex128)

            if use_cache:
                self._trap_cache[f] = vals
            return vals

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
        # compute/retrieve function evaluations
        f_val = self.trap_values(f, k)
        t = np.linspace(0, 1, 2**k + 1)
        dt = t[1] - t[0]

        if df is None:
            # approximate df/dz with finite difference
            dfdt = np.gradient(f_val, dt)
            df_val = dfdt / self.dzdt(t)
        else:
            df_val = self.trap_values(df, k)

        segment_integrand = df_val / f_val * self.dzdt(t)
        if phi is not None:
            segment_integrand = self.trap_values(phi, k) * segment_integrand
        if psi is not None:
            segment_integrand = self.trap_values(psi, k) * segment_integrand

        return scipy.integrate.romb(segment_integrand, dx=dt, axis=-1) / (2j * pi)

    def plot(
        self,
        num_points: int = 100,
        linecolor: Color = "C0",
        linestyle: str = "-",
    ) -> None:
        """
        Uses matplotlib to plot, but not show, the path as a 2D plot in
        the Complex plane.

        Parameters
        ----------
        num_points : int, optional
            The number of points to use when plotting the path.
        linecolor : optional
            The colour of the plotted path, passed to the
            :func:`matplotlib.pyplot.plot` function as the keyword
            argument of 'color'.  See the matplotlib tutorial on specifying
            `colours <https://matplotlib.org/stable/tutorials/colors/colors#>`_.
        linestyle : str, optional
            The line style of the plotted path, passed to the
            :func:`matplotlib.pyplot.plot` function as the keyword
            argument of 'linestyle'.  The default corresponds to a solid
            line.  See :meth:`matplotlib.lines.Line2D.set_linestyle` for
            other acceptable arguments.
        """
        import matplotlib.pyplot as plt

        t = np.linspace(0, 1, num_points)
        path = self(t)
        plt.plot(path.real, path.imag, color=linecolor, linestyle=linestyle)
        plt.xlabel("Re[$z$]", size=16)
        plt.ylabel("Im[$z$]", size=16)
        plt.gca().set_aspect(1)

        # add arrow to indicate direction of path
        arrow_start = self(0.5)
        arrow_end = self(0.51)

        plt.annotate(
            "",
            (arrow_end.real, arrow_end.imag),
            (arrow_start.real, arrow_start.imag),
            arrowprops=dict(arrowstyle="->", fc=linecolor, ec=linecolor),
        )

    def integrate(
        self,
        f: AnalyticFunc,
        abs_tol: float = 1.49e-08,
        rel_tol: float = 1.49e-08,
        div_max: int = 15,
        int_method: IntegrationMethod = "quad",
    ) -> complex:
        r"""
        Integrate the function f along the path.

        .. math::

            \oint_C f(z) dz

        The value of the integral is cached and will be reused if the method
        is called with same arguments.

        Parameters
        ----------
        f : function
            A function of a single complex variable.
        abs_tol : float, optional
            The absolute tolerance for the integration.
        rel_tol : float, optional
            The realative tolerance for the integration.
        div_max : int, optional
            If the Romberg integration method is used then div_max is the
            maximum number of divisions before the Romberg integration
            routine of a path exits.
        int_method : {'quad', 'romb'}, optional
            If 'quad' then :func:`scipy.integrate.quad` is used to
            compute the integral.  If 'romb' then Romberg integraion,
            using :func:`scipy.integrate.romberg`, is used instead.

        Returns
        -------
        complex
            The integral of the function f along the path.
        """

        args = (f, abs_tol, rel_tol, div_max, int_method)
        if args in self._integral_cache.keys():
            return self._integral_cache[args]

        if (
            self._reverse_path is not None
            and args in self._reverse_path._integral_cache
        ):
            # if we have already computed the reverse of this path
            return -self._reverse_path._integral_cache[args]

        @functools.lru_cache(maxsize=None)
        def integrand(t: float) -> complex:
            return f(self(t)) * self.dzdt(t)

        if int_method == "romb":
            integral = scipy.integrate.romberg(
                integrand,
                0,
                1,
                tol=abs_tol,
                rtol=rel_tol,
                divmax=div_max,
            )
        elif int_method == "quad":
            integral = integrate_quad_complex(
                integrand, 0, 1, epsabs=abs_tol, epsrel=rel_tol
            )
        else:
            raise ValueError("int_method must be either 'romb' or 'quad'")

        if np.isnan(integral):
            raise RuntimeError(
                f"The integral along the segment {self} is NaN. This is most "
                "likely due to a root being on or very close to the path of "
                "integration."
            )

        self._integral_cache[args] = integral
        return integral

    def show(self, save_file: Optional[str] = None, **plot_kwargs) -> None:
        """
        Shows the path as a 2D plot in the complex plane.  Requires
        Matplotlib.

        Parameters
        ----------
        save_file : str (optional)
            If given then the plot will be saved to disk with name
            'save_file'.  If save_file=None the plot is shown on-screen.
        **plot_kwargs
            Other key word args are passed to :meth:`~cxroots.Paths.ComplexPath.plot`
        """
        import matplotlib.pyplot as plt

        self.plot(**plot_kwargs)

        if save_file is not None:
            plt.savefig(save_file, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class ComplexLine(ComplexPath):
    r"""
    A straight line :math:`z` in the complex plane from points a to b
    parameterised by

    ..math::

        z(t) = a + (b-a)t, \quad 0\leq t \leq 1
    """

    def __init__(self, a: complex, b: complex):
        self.a, self.b = a, b
        super(ComplexLine, self).__init__()

    def __str__(self):
        return "ComplexLine from %.3f+%.3fi to %.3f+%.3fi" % (
            self.a.real,
            self.a.imag,
            self.b.real,
            self.b.imag,
        )

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
        The function :math:`z(t) = a + (b-a)t`.

        Parameters
        ----------
        t : float
            A real number :math:`0\leq t \leq 1`.

        Returns
        -------
        complex
            A point on the line in the complex plane.
        """
        return self.a + t * (self.b - self.a)

    @overload
    def dzdt(self, t: float) -> complex:
        ...

    @overload
    def dzdt(self, t: npt.NDArray[np.float_]) -> npt.NDArray[np.complex_]:
        ...

    def dzdt(
        self, t: Union[float, npt.NDArray[np.float_]]
    ) -> Union[complex, npt.NDArray[np.complex_]]:
        """
        The derivative of the parameterised curve in the complex plane, z, with
        respect to the parameterization parameter, t.
        """
        return self.b - self.a

    def distance(self, z: complex) -> float:
        """
        Distance from the point z to the closest point on the line.

        Parameters
        ----------
        z : complex

        Returns
        -------
        float
            The distance from z to the point on the line which is
            closest to z.
        """
        # the projection of the point z onto the line a -> b is where
        # the parameter t is
        d = self.b - self.a
        t = ((z - self.a) * d.conjugate()).real / (d * d.conjugate())

        # but the line segment only has 0 <= t <= 1
        t = np.clip(t.real, 0, 1)

        # so the point on the line segment closest to z is
        c = self(t)
        return abs(c - z)


class ComplexArc(ComplexPath):
    r"""
    A circular arc :math:`z` with center z0, radius R, initial angle t0
    and change of angle dt.  The arc is parameterised by

    ..math::

        z(t) = R e^{i(t0 + t dt)} + z0, \quad 0\leq t \leq 1

    Parameters
    ----------
    z0 : complex
    R : float
    t0 : float
    dt : float
    """

    def __init__(self, z0: complex, R: float, t0: float, dt: float):  # noqa: N803
        self.z0, self.R, self.t0, self.dt = z0, R, t0, dt
        super(ComplexArc, self).__init__()

    def __str__(self):
        return "ComplexArc: z0=%.3f, R=%.3f, t0=%.3f, dt=%.3f" % (
            self.z0,
            self.R,
            self.t0,
            self.dt,
        )

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
        The function :math:`z(t) = R e^{i(t_0 + t dt)} + z_0`.

        Parameters
        ----------
        t : float
            A real number :math:`0\leq t \leq 1`.

        Returns
        -------
        complex
            A point on the arc in the complex plane.
        """
        return self.R * np.exp(1j * (self.t0 + t * self.dt)) + self.z0

    @overload
    def dzdt(self, t: float) -> complex:
        ...

    @overload
    def dzdt(self, t: npt.NDArray[np.float_]) -> npt.NDArray[np.complex_]:
        ...

    def dzdt(
        self, t: Union[float, npt.NDArray[np.float_]]
    ) -> Union[complex, npt.NDArray[np.complex_]]:
        """
        The derivative of the parameterised curve in the complex plane, z, with
        respect to the parameterization parameter, t.
        """
        return 1j * self.dt * self.R * np.exp(1j * (self.t0 + t * self.dt))

    def distance(self, z: complex) -> float:
        """
        Distance from the point z to the closest point on the arc.

        Parameters
        ----------
        z : complex

        Returns
        -------
        float
            The distance from z to the point on the arc which is closest
            to z.
        """
        theta = np.angle(z - self.z0)  # np.angle maps to (-pi,pi]
        theta = (theta - self.t0) % (2 * pi) + self.t0  # put theta in [t0,t0+2pi)

        if (self.dt > 0 and self.t0 < theta < self.t0 + self.dt) or (
            self.dt < 0 and self.t0 + self.dt < theta - 2 * pi < self.t0
        ):
            # the closest point to z lies on the arc
            return abs(self.R * np.exp(1j * theta) + self.z0 - z)
        else:
            # the closest point to z is one of the endpoints
            return min(abs(self(0) - z), abs(self(1) - z))
