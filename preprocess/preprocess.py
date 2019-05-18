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

datapoints = N-input_length

X = np.zeros((input_length, datapoints)).astype(int)
Y = np.zeros((output_length, datapoints)).astype(int)

for i in range(datapoints):
	for j in range(input_length):
		X[j, i] = data_indices[i+j]
		Y[0, i] = data_indices[i+j+1]


print(X)
print(Y)

np.save("keras-model/X.npy", X)
np.save("keras-model/Y.npy", Y)