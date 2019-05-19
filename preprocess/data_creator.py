import re
import numpy as np

text = open('data/whole_dataset.txt',encoding='utf8').read().lower()

#text = re.sub(r'[^äö]', r'', text)
text = re.sub(r"[^a-zA-Z0-9äöÄÖåÅ\-,\.?!: \n]", r"", text)
text = text.replace(","," ,")
text = text.replace(":"," :")
text = text.replace("."," .")
text = text.replace("!"," !")
text = text.replace("?"," ?")
text = re.sub(r"\n", r" RIVINVAIHTO ", text)
text = text.split(' ')
N = len(text)

unique_words = []
for w in text:
	if w not in unique_words:
		unique_words.append(w)

words = open("data/distance_data/beginning/words.txt").read().split(";")
wds = []
for w in words:
	wds.append(w.strip())

words = wds

print(len(unique_words), len(words))
print(set(unique_words)-set(words))
print(set(words)-set(unique_words))

text_edited = []
taboos = list(set(words)-set(unique_words))
for w in text:
	if w not in taboos:
		text_edited.append(w)
text = text_edited

ongo = "*klik*" in unique_words
print(ongo)
unique_words = words
unique_words.append("RIVINVAIHTO")

data_indices = np.zeros(N).astype(int)

for i in range(N):
	if text[i] in unique_words:
		index = unique_words.index(text[i])
		data_indices[i] = index
	else:
		print("Word missing... (" + text[i] + ")")

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

