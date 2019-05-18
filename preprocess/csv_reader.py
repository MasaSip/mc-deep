import numpy as np
from numpy import genfromtxt
import os

PATH = "../data/distance_data/beginning/"
TYPE = "distanceMatrix"

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

matrices = []

for filename, number in flist:
	print(filename, number)
	A = genfromtxt(PATH + filename, delimiter=';')
	matrices.append(A)
	print(A.shape)

D = np.concatenate(matrices, axis=0)
print(D.shape)
print(D-D.T)

np.save(PATH + "distance.npy",D)

'''from sklearn.datasets import load_digits
from sklearn.manifold import MDS

embedding = MDS(n_components=12, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(D)

print(X_transformed)

np.save("embedding.npy", X_transformed)
print("saved")
'''
'''
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
svd = TruncatedSVD(n_components=12, n_iter=7, random_state=42)
svd.fit(D[:1000,])
'''
