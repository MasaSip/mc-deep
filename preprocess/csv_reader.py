import numpy as np
from numpy import genfromtxt
import os

PATH = "../data/distance_data/beginning/"
TYPE = "distanceMatrix"

# Find files that are sub-matrices of D
def filename_list(path):
	def myFunc(e):
		return e[1]

	names = list(os.listdir(path))
	tuples = []
	for name in names:
		if TYPE in name:
			print(name)
			number = name.split(".")[0]
			number = number.split(TYPE)[-1]
			tuples.append((name, int(number)))

	tuples.sort(key=myFunc)
	print(tuples)
	return tuples

flist = filename_list(PATH)

# Collect sub-distance matrices into one list
matrices = []
for filename, number in flist:
	print(filename, number)
	A = genfromtxt(PATH + filename, delimiter=';')
	matrices.append(A)
	print(A.shape)

# Combine into one big distance matrix
D = np.concatenate(matrices, axis=0)

# Save into a dedicated numpy array file
np.save(PATH + "distance.npy",D)