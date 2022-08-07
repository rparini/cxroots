from cxroots import AnnulusSector, Rectangle


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
