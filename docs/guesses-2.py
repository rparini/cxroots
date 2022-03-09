from cxroots import Rectangle
C = Rectangle([-1.5,1.5], [-1.5,1.5])
f  = lambda z: z**26-2*z**10+0.5*z**6-1
df = lambda z: 26*z**25-20*z**9+3*z**5
C.demo_roots(f, df, guessRootSymmetry=lambda z: [z.conjugate(), -z], saveFile='ex_rootSymmetry.gif', writer='imagemagick')