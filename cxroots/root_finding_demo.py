import numpy as np

from .root_finding import find_roots_gen


def demo_find_roots(
    original_contour,
    f,
    df=None,
    save_file=None,
    auto_animation=False,
    return_animation=False,
    writer=None,
    **roots_kwargs
):
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
    return_animation : bool, optional
        If True then the matplotlib animation object will be returned
        instead of being shown.  Defaults to False.
    writer : str, optional
        Passed to :meth:`matplotlib.animation.FuncAnimation.save`.
    **roots_kwargs
        Additional key word arguments passed to :meth:`~cxroots.contour.Contour.roots`.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.gcf()
    ax = plt.gca()

    root_finder = find_roots_gen(original_contour, f, df, **roots_kwargs)

    def init():
        original_contour.plot(linecolor="k", linestyle="--")
        original_contour._size_plot()

    def update_frame(args):
        roots, _, boxes, num_remaining_roots = args

        plt.cla()  # clear axis
        original_contour.plot(linecolor="k", linestyle="--")
        for box in boxes:
            if not hasattr(box, "_color"):
                cmap = plt.get_cmap("jet")
                box._color = cmap(np.random.random())

            plt.text(box.central_point.real, box.central_point.imag, box._num_roots)
            box.plot(linecolor=box._color)

        plt.scatter(np.real(roots), np.imag(roots), color="k", marker="x")
        ax.text(
            0.02,
            0.95,
            "Zeros remaining: %i" % num_remaining_roots,
            transform=ax.transAxes,
        )
        original_contour._size_plot()
        fig.canvas.draw()

    if save_file or auto_animation or return_animation:
        anim = animation.FuncAnimation(
            fig, update_frame, init_func=init, frames=root_finder
        )

    if return_animation:
        return anim
    elif save_file:
        anim.save(filename=save_file, fps=1, dpi=200, writer=writer)
        plt.close()
    elif auto_animation:
        plt.show()
    else:
        # Create event to handler to let user move through frames

        def draw_next(event):
            if event.key == " ":
                update_frame(next(root_finder))

        fig.canvas.mpl_connect("key_press_event", draw_next)
        plt.show()

    return None
