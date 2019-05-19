from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
from fbpca import diffsnorm, pca
import math

embedding_type = "end"
PATH = "../data/distance_data/" + embedding_type +"/"
MAX_DIM = 30000
MAX_DIM_Y = 10000
D = np.load(PATH + "distance.npy")
print("Distance matrix loaded")
D_small = D[:MAX_DIM, :MAX_DIM_Y]
print("Normalize")
D_small = D_small / np.max(D_small)
print("Nonlinearize")
# A non-linearity is used to emphasize words with high similarity
D_small = -(D_small ** 0.25) +1
D_small = D_small

print("Calculate SVD...")
U, s, Va = pca(D_small, k=64, raw=True, n_iter=15, l=None)
print(U)
print("DONE!")

err = diffsnorm(D_small, U, s, Va)
print(err)

print("Print embedding vector lengths...")
for i in range(len(U)):
	print(np.sum(U[i,:] * U[i,:]) ** 0.5)


I = np.identity(len(D_small))[:len(D_small), :len(D_small[0])]
D_nondiagonal = D_small + I * 8.0

min_index = np.unravel_index(D_nondiagonal.argmin(), D_nondiagonal.shape)
print(min_index, D[min_index[0], min_index[1]])

print("Load word list...")
f = open(PATH + "words.txt", "r").read().split(";")

words = list(f)
'''
for x in words[470:480]:
	print(x)

print(words[min_index[0]], words[min_index[1]])
print(words[50], words[51], D[50,51])

print(U[51], U[50])
'''

print("Normalize embedding to unit length...")
for i in range(len(U)):
	U[i,:] = U[i,:] / ( np.sum(U[i,:] * U[i,:]) ** 0.5)
print("Done!")



test_1 = False
test_2 = False
test_3 = False
test_n = test_1 or test_2 or test_3

print("Run any tests:", test_n)
if test_n:
	print("Calculate distance matrix of embedding...")
	from scipy.spatial import distance_matrix
	D_approximate = distance_matrix(U, U)
	print("Done!")
	minmatr = D_approximate+ I * 8.0
	print(np.min(D_approximate+ I * 8.0), np.max(D_approximate))


print("Run Test 1:", test_1)
if test_1:
	for i in range(5000):
		min_index = np.unravel_index(minmatr.argmin(), minmatr.shape)
		print(words[min_index[0]], words[min_index[1]], minmatr[min_index[0], min_index[1]])
		minmatr[min_index[0], min_index[1]] = 8.0
		minmatr[min_index[1], min_index[0]] = 8.0

print("Run Test 2:", test_2)
if test_2:
	for i in range(len(D_approximate)):
		A = minmatr[i,:]
		vec = U[i,:]
		min_index = A.argmin()
		max_index = vec.argmax()
		print(words[i], words[min_index], A[min_index])
		#print(words[i], words[max_index])
		#minmatr[min_index[0], min_index[1]] = 8.0
		#minmatr[min_index[1], min_index[0]] = 8.0

def angle(distance):
	r = math.acos(1- distance/2)
	return math.degrees(r)

print("Run Test 3:", test_3)
if test_3:
	l = list(minmatr.flatten())
	l2 = []

	under = 0
	for elem in l:
		if elem == 8.0:
			0
		else:
			degrees = angle(elem)
			l2.append(degrees)
			if degrees < 45:
				under += 1

	import matplotlib.pyplot as plt

	plt.hist(l2, bins=200)
	#plt.yscale('log')
	print("Under 45 degrees:", under)
	plt.show()

print("Saving matrix of size", U.shape,"and type","'" + embedding_type+ "' as embedding...")
np.save("../data/embeddings/embedding-" + embedding_type +".npy", U)
print("DONE!")
