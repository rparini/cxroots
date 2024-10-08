from typing import NamedTuple

import numpy as np

from cxroots.contour_interface import ContourABC


class RootResult(
    NamedTuple("RootResult", [("roots", list[complex]), ("multiplicities", list[int])])
):
    """
    A class which stores the roots and their multiplicites as attributes
    and provides convienent methods for displaying them.

    Attributes
    ----------
    roots : list
        List of roots
    multiplicities : list
        List of multiplicities where the ith element of the list is the
        multiplicity of the ith element of roots.
    contour : Contour
        The contour bounding the region in which the roots were found.
    """

    def __new__(
        cls, roots: list[complex], multiplicities: list[int], contour: ContourABC
    ):
        return super().__new__(cls, roots, multiplicities)

    def __init__(
        self, roots: list[complex], multiplicities: list[int], contour: ContourABC
    ):
        self.contour = contour
        super().__init__()

    def show(self, save_file: str | None = None) -> None:
        """
        Plot the roots and the initial integration contour in the
        complex plane.

        Parameters
        ----------
        save_file : str, optional
            If provided the plot of the roots will be saved with
            file name save_file instead of being shown.

        Example
        -------
        .. plot::
            :include-source:

            from cxroots import Circle
            C = Circle(0, 2)
            f = lambda z: z**6 + z**3
            df = lambda z: 6*z**5 + 3*z**2
            r = C.roots(f, df)
            r.show()
        """
        import matplotlib.pyplot as plt

        self.contour.plot(linecolor="k", linestyle="--")
        plt.scatter(np.real(self.roots), np.imag(self.roots), color="k", marker="x")

        if save_file is not None:
            plt.savefig(save_file)
            plt.close()
        else:
            plt.show()

    def __str__(self):
        roots, multiplicities = np.array(self.roots), np.array(self.multiplicities)

        # reorder roots
        sortargs = np.argsort(roots)
        roots, multiplicities = roots[sortargs], multiplicities[sortargs]

        s = " Multiplicity |               Root              "
        s += "\n------------------------------------------------"

        for i, root in np.ndenumerate(roots):
            if root.real < 0:
                s += (
                    f"\n{int(multiplicities[i]): ^14d}| "
                    f"{root.real:.12f} {root.imag:+.12f}i"
                )
            else:
                s += (
                    f"\n{int(multiplicities[i]): ^14d}|  "
                    f"{root.real:.12f} {root.imag:+.12f}i"
                )

        return s
