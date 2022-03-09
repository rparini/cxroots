from cxroots import Circle
from numpy import exp, cos, sin
f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)
C = Circle(0,3)

C.demo_roots(f, saveFile='rootsDemo.gif', writer='imagemagick')