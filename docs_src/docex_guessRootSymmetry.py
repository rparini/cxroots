from cxroots import Rectangle
C = Rectangle([-1.5,1.5], [-1.5,1.5])
f  = lambda z: z**26-2*z**10+0.5*z**6-1
df = lambda z: 26*z**25-20*z**9+3*z**5

rootSymmetry = lambda z: [z.conjugate(), -z]
C.demo_roots(f, df, guessRootSymmetry=rootSymmetry, M=1, saveFile='ex_rootSymmetry.gif', writer='imagemagick')

# Using guessRootSymmetry can same some time:
from time import time
t0 = time()
C.roots(f, df)
t1 = time()
C.roots(f, df, guessRootSymmetry = rootSymmetry)
t2 = time()

print('Time without symmetry:', t1-t0)
print('Time with symmetry:', t2-t1)
