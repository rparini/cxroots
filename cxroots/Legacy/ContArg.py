import numpy as np
import matplotlib.pyplot as plt
from scipy import pi, exp

def contArg(z):
	# return the continued argument of the list of point z
	arg = np.angle(z)
	posIndices = np.where(np.abs(arg[1:] - arg[:-1]) > np.abs(arg[1:] - arg[:-1] + 2*pi))[0]
	for index in posIndices:
		arg[index+1:] += 2*pi

	negIndices = np.where(np.abs(arg[1:] - arg[:-1]) > np.abs(arg[1:] - arg[:-1] - 2*pi))[0]
	for index in negIndices:
		arg[index+1:] -= 2*pi

	return arg
