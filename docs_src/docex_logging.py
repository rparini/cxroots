import logging
from numpy import exp, cos, sin
from cxroots import Circle

C = Circle(0, 3)
f = lambda z: (exp(-z) * sin(z / 2) - 1.2 * cos(z)) * (z + 1.2) * (z - 2.5) ** 2

# use logging.DEBUG to show diagnostic information
logging.basicConfig(level=logging.INFO) 
C.roots(f)
