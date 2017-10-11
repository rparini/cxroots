from numpy import exp, sin, cos
from cxroots import Circle
C = Circle(0, 3)
f = lambda z: (z-2.5)**2 * (z+1.2) * (exp(-z)*sin(z/2)-1.2*cos(z))

roots = C.roots(f, guessRoots=[(2.5,2), (-1.2,1)])
print(roots)

C.demo_roots(f, guessRoots=[(2.5,2), (-1.2,1)], M=1, saveFile='guessRoots.gif', writer='imagemagick')


# animation = C.demo_roots(f, guessRoots=[(2.5,2)], M=1, returnAnim=True)
# from IPython.display import HTML
# print(HTML(animation.to_html5_video()))