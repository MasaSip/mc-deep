from custom_levenshtein import levenshtein as distance
import numpy as np

rhyme = ""

with open('heikkikuula.txt', 'r') as file:
    rhyme = file.read().replace('\n', ' ').lower()

r = list(set(rhyme.split()))

smallest_d = 1000.0
smallest = ""

r = list(np.array(r)[:400])
N = len(r)
print("N", N)
D = np.zeros((N,N))
for i, w1 in enumerate(r):
	if i % 100 == 0:
		print(i)
	for j, w2 in enumerate(r):
		d = distance(w1,w2, reverse=True, exp=True)

		D[i,j] = d
		if d > 0.0 and d < smallest_d:
			smallest_d = d
			smallest = w1 + " " + w2

print(smallest, smallest_d)

EPSILON = 0.01
DELTA = 0.00000001

absD = np.abs(D-D.T)

Ddelta = D + DELTA
print(D-D.T, np.count_nonzero(absD / Ddelta < EPSILON),
	np.count_nonzero(absD / Ddelta >= EPSILON))

from sklearn.datasets import load_digits
from sklearn.manifold import MDS

DIMENSIONS = 2
embedding = MDS(n_components=DIMENSIONS, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(D)

print(X_transformed)

print(r)

import matplotlib.pyplot as plt
plt.ion()

y = list(X_transformed[:,0])
z = list(X_transformed[:,1])
n = r

fig, ax = plt.subplots()
ax.scatter(y, z)

for i, txt in enumerate(n):
    ax.annotate(txt, (y[i], z[i]))

while True:
	testword = input("Word to embed : ") 

	distances = []
	for i, w1 in enumerate(r):
		d = distance(w1,testword, reverse=True, exp=True)
		distances.append(d)

	K = 3
	k_closest = np.argsort(distances)[:5]

	k_weights = []
	for i in k_closest:
		d = distances[i]
		w = r[i]
		k_weights.append(1/(d ** 2))
		print(w, d)

	k_weights = np.array(k_weights)
	k_weights = list(k_weights / np.sum(k_weights))

	coords_0 = np.zeros(DIMENSIONS)
	for i, weight in enumerate(k_weights):
		j = k_closest[i]
		print(r[j], weight)
		coords = X_transformed[j,:]
		coords_0 = coords_0 + coords*weight

	print(coords_0)

	a = [coords_0[0]]
	b = [coords_0[1]]

	ax.annotate(testword, (a[0], b[0]))

	ax.scatter(a, b, color="red")

	plt.show()
