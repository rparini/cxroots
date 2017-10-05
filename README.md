
# cxroots
A Python module to compute all the roots of an analytic function <img src="https://rawgit.com/RParini/cxroots/multiplicities/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.970500000000005pt height=24.65759999999998pt/> within a given contour <img src="https://rawgit.com/RParini/cxroots/multiplicities/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.924780000000005pt height=22.46574pt/> provided that <img src="https://rawgit.com/RParini/cxroots/multiplicities/svgs/210d22201f1dd53994dc748e91210664.svg?invert_in_darkmode" align=middle width=30.970500000000005pt height=24.65759999999998pt/>:

* has no roots or poles on <img src="https://rawgit.com/RParini/cxroots/multiplicities/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.924780000000005pt height=22.46574pt/>
* is analytic in the interior of <img src="https://rawgit.com/RParini/cxroots/multiplicities/svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.924780000000005pt height=22.46574pt/>

The implementation is based on the method in [1].


```python
from cxroots import Circle
from numpy import exp, cos, sin

f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)

C = Circle(0,3)
C.print_roots(f)
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



```python
C.show_roots(f)
```


![png](README_files/README_2_0.png)


### Installation
To install cxroots click the green download button above and then from the command line run the included 'setup.py' using
```bash
python setup.py install
```

### Documentation
For a tutorial on the use of this module and a discrption of how it works see the [documentation](https://rparini.github.io/cxroots/).

### See also
The Fortran 90 package [ZEAL](http://cpc.cs.qub.ac.uk/summaries/ADKW_v1_0.html) is a direct implementation of [1].

---

#### References
[1] Kravanja, Peter, and Marc Van Barel. "Zeros of analytic functions." Computing the Zeros of Analytic Functions. Springer Berlin Heidelberg, 2000. 1-59.

