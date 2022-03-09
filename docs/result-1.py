from cxroots import Circle
C = Circle(0, 2)
f = lambda z: z**6 + z**3
df = lambda z: 6*z**5 + 3*z**2
r = C.roots(f, df)
r.show()