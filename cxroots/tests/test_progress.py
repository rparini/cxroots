from cxroots import Circle


def test_progress():
    """Test progress bar (verbose=True) works without error"""
    contour = Circle(0, 3)

    def f(z):
        return (z + 1.2) ** 3 * (z - 2.5) ** 2 * (z + 1j)

    contour.roots(f, verbose=True)
