## Changelog
### Unreleased
- Replaced docrep with numpydoc

### 1.1.9 - 1/Mar/2020
- Fix depreciation warning from scipy

### 1.1.8 - 20/Apr/2019
- Fix NotImplementedError in Contour base ([#31](https://github.com/rparini/cxroots/issues/31) thanks [@fchapoton](https://github.com/fchapoton))
- Added `__call__` placeholder to ComplexPath

### 1.1.7 - 10/Feb/2019
- Fix `setup.py test`.
- Fix invalid escape characters

### 1.1.6 - 19/Sep/2018
- Added example to index page of documentation.
- Removed unnecessary imports.
- Enable `setup.py test`.

### 1.1.5 - 5/Aug/2018
- Got project description working on PyPi.

### 1.1.0 - 5/Aug/2018
- New tests and converted old tests to pytest.
- Documentation overhauled to include separate tutorial, user guide and reference sections.  In the process added/improved many docstrings.
- Function capitalisation now consistent.
- Changed AnnulusSector's rRange -> radii to match Annulus.
- Contours which have some known roots and unknown roots in them will be subdivided to avoid calculating already known roots.
- If the number of roots (counting multiplicities) in a contour does not match the sum of the number of roots in its subdivided contours then the number of roots is recalculated in both the parent and child contours with NIntAbsTol set to half the provided value.
- Romberg integration now requires the last two consecutive iterations (instead of just the last) to satisfy the prescribed error tolerance before the last iteration is accepted as the result.
- If Romberg integration is used then the function evaluations are cached and the number of roots within a contour checked while the roots within the contour are being calculated.
- Introduced parameter m which is only used if df=None and method='quad'.  The argument order=m is the order of the error term in the approximation for the derivative used during contour integration.
- Implemented methods to determine the distance between a point and the closest point on a line or circular arc in the complex plane.
- If a proposed subdivision introduces a contour with a minimum distance of < 0.01 from any known root then it is discarded and a new subdivision chosen.
- Exposes the parameter errStop to the user.  The number of distinct roots within a contour, n, is determined by checking if all the elements of a list of contour integrals involving formal orthogonal polynomials are sufficently close to zero, ie. that the absolute value of each element is < errStop.  If errStop is too large/small then n may be smaller/larger than it actually is.
- Computing multiplicities now reuses integrals used to calculate distinct roots.
- If the computed multiplicity for a root is not sufficiently close to an integer (as determined by integerTol) then the multiplicity will be determined again by computing the roots within a small circle centred on the root.
- Final multiplicities are now returned as integers, rather than rounded floats.
- Muller’s method rather than Newton-Raphson iteration if df is not provided.
- Added attemptIterBest parameter.  If True then the iterative method used to refine the roots will exit when error of the previous iteration, x0, was at least as good as the current iteration, x, in the sense that abs(f(x)) >= abs(f(x0)) and the previous iteration satisfied abs(dx0) < newtonStepTol.  In this case the previous iteration is returned as the approximation of the root.  The idea is that it will attempt to get the best accuracy it can at the expense of more function evaluations.
- RootResult now a subclass of namedtuple()


### 1.0.0 - 12/Oct/2017
Initial release version.
