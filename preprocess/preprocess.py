import re
import numpy as np

text = open('demos/embedding_demo/heikkikuula.txt').read()

#text = re.sub(r'[^äö]', r'', text)
text = re.sub(r"[^a-zA-Z \n]", r"", text)
text = re.sub(r"\n", r" RIVINVAIHTO ", text)
text = text.split(' ')
N = len(text)

unique_words = []
for w in text:
	if w not in unique_words:
		unique_words.append(w)

data_indices = np.zeros(N).astype(int)

for i in range(N):
	index = unique_words.index(text[i])
	data_indices[i] = index

print(data_indices)
print(np.max(data_indices))

input_length = 20
output_length = 1
samples = N-input_length
features = 1

X = np.zeros((samples, input_length, features)).astype(int)
Y = np.zeros(samples).astype(int)

for i in range(samples):
	for j in range(input_length):
		X[i, j, 0] = data_indices[i+j]
	category = data_indices[i+input_length]
	Y[i] = data_indices[i+input_length]


print(X)
print(Y)

np.save("keras-model/X.npy", X)
np.save("keras-model/Y.npy", Y)

U = np.array(unique_words)
np.save("keras-model/U.npy", U)