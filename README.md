
# cxroots
A Python module to compute all the (simple) roots of an analytic function within a given contour.

---

## Introduction

Given a contour <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/> in the complex plane and a function <img src="https://rawgit.com/RParini/cxroots/None/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.864075pt height=24.56552999999997pt/> (and optionally it's derivative <img src="https://rawgit.com/RParini/cxroots/None/svgs/fc05b2681daad9671bb48c269dcca2d6.svg?invert_in_darkmode" align=middle width=35.47599pt height=24.668490000000013pt/>) which:

* Has no roots or poles on <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>
* Is analytic in the interior of <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>
* Has only simple roots in the interior of <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>

the module is able to compute all the roots of within <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>.


```python
from cxroots import Rectangle, showRoots, findRoots, demo_findRoots
from numpy import sin, cos

rect = Rectangle([-1.5,1.5],[-2,2])
f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
showRoots(rect, f)
```


![png](readme_input_files/readme_input_1_0.png)


The implementation is primarily based on [1] where the number of roots within a contour, <img src="https://rawgit.com/RParini/cxroots/None/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.944050000000002pt height=22.381919999999983pt/>, is calculated by numerical integration of the Cauchy integral,

<p align="center"><img src="https://rawgit.com/RParini/cxroots/None/svgs/366e872f6f14ca3dcb6c05bd9058dc91.svg?invert_in_darkmode" align=middle width=151.216065pt height=38.824995pt/></p>

The original contour is subdivided until each sub-contour only contains a single root and then the Newton-Raphson method is repeatedly used with random startpoints until the root within each sub-contour is found.

If <img src="https://rawgit.com/RParini/cxroots/None/svgs/fc05b2681daad9671bb48c269dcca2d6.svg?invert_in_darkmode" align=middle width=35.47599pt height=24.668490000000013pt/> is not provided by the user then it is approximated with a Taylor expansion as in [2].

### Future improvements:
* Allow for multiplicities of roots within <img src="https://rawgit.com/RParini/cxroots/None/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>.  Perhaps using the method of formal orthogonal polynomials suggested in [3]
* Approximate the roots of <img src="https://rawgit.com/RParini/cxroots/None/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705000000003pt height=22.745910000000016pt/> within a contour as the zeros of a constructed polynomial [2]
* If <img src="https://rawgit.com/RParini/cxroots/None/svgs/fc05b2681daad9671bb48c269dcca2d6.svg?invert_in_darkmode" align=middle width=35.47599pt height=24.668490000000013pt/> is not provided then use the approximation to <img src="https://rawgit.com/RParini/cxroots/None/svgs/fc05b2681daad9671bb48c269dcca2d6.svg?invert_in_darkmode" align=middle width=35.47599pt height=24.668490000000013pt/> in the Newton-Raphson method

### Installation
Can be downloaded from GitHub and installed by running the included 'setup.py' with
```bash
python setup.py install
```

### Documentation
For a tutorial on the use of this module and a discrption of how it works see the [documentation](https://rparini.github.io/cxroots/).

---

#### References
[1] M. Dellnitz, O. Schütze and Q. Zheng, "Locating all the Zeros of an Analytic Function in one Complex Variable" J. Compu. and App. Math. (2002) Vol. 138, Issue 2

[2] L.M. Delves and J.N. Lyness, "A Numerical Method for Locating the Zeros of an Analytic function" Mathematics of Computation (1967) Vol.21, Issue 100

[3] P. Kravanja, T. Sakurai and M. Van Barel, "On locating clusters of zeros of analytic functions" BIT (1999) Vol. 39, No. 4
