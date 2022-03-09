from numpy import exp, sin, cos
from cxroots import Circle
C = Circle(0, 3)
f = lambda z: (exp(-z)*sin(z/2)-1.2*cos(z))*(z+1.2)*(z-2.5)**2
C.demo_roots(f, guessRoots=[2.5, -1.2], saveFile='guessRoots.gif', writer='imagemagick')