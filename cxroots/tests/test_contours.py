import numpy as np

from cxroots import AnnulusSector, Circle, Rectangle


def test_annulus_sector_contains():
    r0 = 8.938
    r1 = 9.625
    phi0 = 6.126
    phi1 = 6.519

    z = 9 - 1.04825594683e-18j
    contour = AnnulusSector(0, [r0, r1], [phi0, phi1])

    assert contour.contains(z)


def test_rect_contains():
    contour = Rectangle([-2355, -1860], [-8810, -8616])

    assert contour.contains(-2258 - 8694j)
    assert not contour.contains(-2258 - 8500j)


def test_rect_call():
    c = Rectangle([0, 2], [0, 1])

    t_arr = np.array([0, 0.125, 0.25, 0.5, 0.75, 1])
    z_arr = np.array([0, 1, 2, 2 + 1j, 1j, 0])

    # Test passing in the whole array
    assert np.all(c(t_arr) == z_arr)

    # Test individual calls
    for t, z in zip(t_arr, z_arr):
        assert c(t) == z


def test_circle_call():
    c = Circle(0, 1)

    t_arr = np.array([0, 0.25])
    z_arr = np.array([1, 6.123233995736766e-17 + 1j])

    # Test passing in the whole array
    assert np.all(c(t_arr) == z_arr)

    # Test individual calls
    for t, z in zip(t_arr, z_arr):
        assert c(t) == z
