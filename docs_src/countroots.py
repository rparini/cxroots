from cxroots import Circle
from numpy import exp, cos, sin
f = lambda z: (z*(z+2)*(z+4))**2
C = Circle(0,3)

num_roots = C.count_roots(f)
print(num_roots)
