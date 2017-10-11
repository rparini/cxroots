from numpy import sin, cos 			
f  = lambda z: 1j*z**5 + z*sin(z)          # Define the function that we want to find the roots of
df = lambda z: 5j*z**4 + z*cos(z) + sin(z) # Define the corresponding derivative (optional but helpful)

from cxroots import Circle 			
C = Circle(0, 2)		# Define a circle, centered at 0 and with radius 2
r = C.roots(f, df)		# Find the roots of f(z) within the circle 