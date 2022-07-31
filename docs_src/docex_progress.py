from cxroots import Circle

C = Circle(0, 3)
f = lambda z: (z + 1.2) ** 3 * (z - 2.5) ** 2 * (z + 1j)
C.roots(f, verbose=True)
