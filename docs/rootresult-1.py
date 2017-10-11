from cxroots import Circle
C = Circle(0, 2)
f = lambda z: z**6 + z**3
r = C.roots(f)
r.show()