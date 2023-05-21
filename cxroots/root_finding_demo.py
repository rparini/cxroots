from typing import Optional

import numpy as np

from cxroots.types import AnalyticFunc

from .contour_interface import ContourABC
from .root_finding import RootFinderState, find_roots_gen


def _contour_color(contour: ContourABC):
    """
    Deterministically generate a colour for a contour so that the contour has the same
    colour in each frame of the root finding animation
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")

    rng = np.random.default_rng(hash(contour))
    return cmap(rng.random())


def _update_frame(frame: RootFinderState, original_contour: ContourABC) -> None:
    import matplotlib.pyplot as plt

    ax = plt.gca()

    roots, _, contour_data, num_remaining_roots = frame

    plt.cla()  # clear axis
    original_contour.plot(linecolor="k", linestyle="--")
    for data in contour_data:
        contour = data.contour
        plt.text(
            contour.central_point.real, contour.central_point.imag, str(data.num_roots)
        )
        contour.plot(linecolor=_contour_color(contour))

    plt.scatter(np.real(roots), np.imag(roots), color="k", marker="x")
    ax.text(
        0.02, 0.95, "Zeros remaining: %i" % num_remaining_roots, transform=ax.transAxes
    )
    original_contour.size_plot()
    plt.draw()


def demo_roots_animation(
    original_contour: ContourABC,
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    **roots_kwargs
):
    """
    Creates an animation object where each frame is a step in the rootfinding process

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
    **roots_kwargs : kwargs
        Other keyword arguments to pass to
        :func:`find_roots_gen <cxroots.root_finding.find_roots_gen>`

    Returns
    -------
    animation.FuncAnimation
        An animation where each frame is a step in the rootfinding process
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.gcf()

    root_finder = find_roots_gen(original_contour, f, df, **roots_kwargs)
    return animation.FuncAnimation(
        fig,
        _update_frame,
        frames=root_finder,
        fargs=[original_contour],
        cache_frame_data=False,
    )


def demo_find_roots(
    original_contour: ContourABC,
    f: AnalyticFunc,
    df: Optional[AnalyticFunc] = None,
    save_file: Optional[str] = None,
    auto_animation: bool = False,
    writer: Optional[str] = None,
    **roots_kwargs
) -> None:
    """
    An animated demonstration of the root finding process using matplotlib.

    Parameters
    ----------
    save_file : str, optional
        If given then the animation will be saved to disk with filename
        equal to save_file instead of being shown.
    auto_animation : bool, optional
        If False (default) then press SPACE to step the animation forward
        If True then the animation will play automatically until all the
        roots have been found.
    writer : str, optional
        Passed to :meth:`matplotlib.animation.FuncAnimation.save`.
    **roots_kwargs
        Additional key word arguments passed to :meth:`~cxroots.contour.Contour.roots`.
    """
    import matplotlib.pyplot as plt

    if save_file or auto_animation:
        anim = demo_roots_animation(original_contour, f, df=None, **roots_kwargs)
        if save_file:
            anim.save(filename=save_file, fps=1, dpi=200, writer=writer)
            plt.close()
        else:
            plt.show()

    else:
        # Create event handler to let user move through frames
        root_finder = find_roots_gen(original_contour, f, df, **roots_kwargs)
        original_contour.plot(linecolor="k", linestyle="--")

        def draw_next(event):
            if event.key == " ":
                try:
                    _update_frame(next(root_finder), original_contour)
                except StopIteration:
                    # No more roots to find
                    pass

        fig = plt.gcf()
        fig.canvas.mpl_connect("key_press_event", draw_next)  # type: ignore
        plt.show()
