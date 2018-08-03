from numpy import exp, cos, sin
f = lambda z: (exp(2*z)*cos(z)-1-sin(z)+z**5)*(z*(z+2))**2

from cxroots import Circle
C = Circle(0,3)
roots = C.roots(f)
roots.show('readmeEx.png')

print(roots)