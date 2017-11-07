
![](https://travis-ci.org/rparini/cxroots.svg?branch=master)
# cxroots
A Python module to compute all the roots of a function <img src="https://rawgit.com/RParini/cxroots/dev/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.864075pt height=24.56552999999997pt/> of a single complex variable which lie within a given contour <img src="https://rawgit.com/RParini/cxroots/dev/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>.  The function <img src="https://rawgit.com/RParini/cxroots/dev/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.864075pt height=24.56552999999997pt/> must:

* have no roots or poles on <img src="https://rawgit.com/RParini/cxroots/dev/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>
* be analytic in the interior of <img src="https://rawgit.com/RParini/cxroots/dev/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435000000003pt height=22.381919999999983pt/>

The implementation uses the method in [1] to approximate the roots within a contour by evaluating integrals of the form,

<p align="center"><img src="https://rawgit.com/RParini/cxroots/dev/svgs/961e481665f16a3c8067223dbb40f1d7.svg?invert_in_darkmode" align=middle width=138.70593pt height=38.824995pt/></p>

for certain polynomials <img src="https://rawgit.com/RParini/cxroots/dev/svgs/919032a59e522724a0465e9dad600828.svg?invert_in_darkmode" align=middle width=30.834705000000003pt height=24.56552999999997pt/>.  If <img src="https://rawgit.com/RParini/cxroots/dev/svgs/fc05b2681daad9671bb48c269dcca2d6.svg?invert_in_darkmode" align=middle width=35.47599pt height=24.668490000000013pt/> is not provided by the user then it is approximated using a finite difference method.  The approximations to the roots are then refined using the Newton-Raphson or Muller's (if only <img src="https://rawgit.com/RParini/cxroots/dev/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.864075pt height=24.56552999999997pt/> if given) method.

### Documentation
For more information and a tutorial on cxroots see the [documentation](https://rparini.github.io/cxroots/).


```python
from numpy import exp, cos, sin
f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)

from cxroots import Circle
C = Circle(0,3)
roots = C.roots(f)
roots.show()
```


![png](README_files/README_1_0.png)



```python
print(roots)
```

     Multiplicity |               Root              
    ------------------------------------------------
          2       | -2.000000000000 -0.000000000000i
          1       | -0.651114070264 -0.390425719088i
          1       | -0.651114070264 +0.390425719088i
          3       |  0.000000000000 +0.000000000000i
          1       |  0.648578080954 -1.356622683988i
          1       |  0.648578080954 +1.356622683988i
          1       |  2.237557782467 +0.000000000000i


### Installation
cxroots requires [Python](https://www.python.org/downloads/).  Python 2 and 3 are both compatible.

The easiest way to install cxroots with the Python Package Index by entering on the command line:
```bash
pip install cxroots
```

### See also
The Fortran 90 package [ZEAL](http://cpc.cs.qub.ac.uk/summaries/ADKW_v1_0.html) is a direct implementation of [1].

---

#### References
[1] Kravanja, Peter, and Marc Van Barel. "Zeros of analytic functions." Computing the Zeros of Analytic Functions. Springer Berlin Heidelberg, 2000. 1-59.

