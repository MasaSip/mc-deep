from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
from fbpca import diffsnorm, pca

PATH = "../data/distance_data/beginning/"
MAX_DIM = 5000
D = np.load(PATH + "distance.npy")
D_small = D[:MAX_DIM, :MAX_DIM]

U, s, Va = pca(D_small, k=32, raw=True, n_iter=15, l=None)
print(U)

err = diffsnorm(D_small, U, s, Va)
print(err)

for i in range(MAX_DIM):
	print(np.sum(U[i,:] * U[i,:]) ** 0.5)


D_nondiagonal = D_small + np.identity(MAX_DIM) * 8.0

min_index = np.unravel_index(D_nondiagonal.argmin(), D_nondiagonal.shape)
print(min_index, D[min_index[0], min_index[1]])

f = open(PATH + "words.txt", "r").read().split(";")

words = list(f)

for x in words[470:480]:
	print(x)

print(words[min_index[0]], words[min_index[1]])
print(words[50], words[51], D[50,51])