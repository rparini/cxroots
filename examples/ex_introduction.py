from cxroots import Rectangle
from numpy import sin, cos

f = lambda z: (sin(z)*cos(z/2)-2*z**5)*(z**2+1)*(z**2-1)**2

rect = Rectangle([-1.5,1.5],[-2,2])
rect.print_roots(f)
