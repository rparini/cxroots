from numpy import exp, sin, cos
from cxroots import Circle
C = Circle(0, 3)
f = lambda z: (exp(-z)*sin(z/2)-1.2*cos(z))*(z+1.2)*(z-2.5)**2
C.demo_roots(f, guess_roots=[2.5, -1.2], save_file='guess_roots.gif', writer='imagemagick')