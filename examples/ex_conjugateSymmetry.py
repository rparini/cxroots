from cxroots import Circle

C = Circle(0, 1.5)
f = lambda z: z**27-2*z**11+0.5*z**6-1
df = lambda z: 27*z**26-22*z**10+3*z**5

# Since f is a polynomial with real coefficients we know that any
# complex roots must come in conjugate pairs and telling cxroots
# this will save some effort.

# This is done using the guessRootSymmetry argument which should be
# a function of a complex number, z, which returns a list of roots
# assuming that z is a root.

# So for example here:
conjugateSymmetry = lambda z: [z.conjugate()]
roots, multiplicities = C.roots(f, df, guessRootSymmetry = conjugateSymmetry)

# The rootfinding process can be saved to a video file:
C.demo_roots(f, df, guessRootSymmetry=conjugateSymmetry, saveFile='ex_conjugateSymmetry.mov')

# Using guessRootSymmetry can same some time:
from time import time
t0 = time()
C.roots(f, df)
t1 = time()
C.roots(f, df, guessRootSymmetry = conjugateSymmetry)
t2 = time()

print('Time without symmetry:', t1-t0)
print('Time with symmetry:', t2-t1)
