from numpy import exp
from cxroots import Rectangle

A = -0.19435
B = 1000.41
C = 522463
T = 0.005

f = lambda z: z**2 + A*z + B*exp(-T*z) + C
df = lambda z: 2*z + A - B*T*exp(-T*z)

rectangle = Rectangle([-15000,5000], [-15000,15000])
roots = rectangle.roots(f, df, rootErrTol=1e-6)
roots.show()