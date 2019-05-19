from custom_levenshtein import levenshtein

filename = "sample_output.txt"
f = open(filename, "r").readlines()

last_words = []

for line in f:
	s = line.split()
	if len(s) > 0:
		last = line.split()[-1]
		last_words.append(last)

distances = []

for i in range(len(last_words)-1):
	w1 = last_words[i]
	w2 = last_words[i+1]

	dist = levenshtein(w1, w2, True, True)
	print(dist)
	distances.append(dist)

import numpy as np
distances = np.array(distances)


print("MEAN", np.mean(distances))


# MEAN OF CONSECUTIVE LAST WORDS 0.9876374414207171

# MEAN OF OUR MODEL 1.5247472091179042
# MEAN OF ALL 1.4579899821773632
