from custom_levenshtein import levenshtein
import numpy as np

filename = "random_embedding.txt"
f = open(filename, "r").readlines()

# Find last words of each row
last_words = []
for line in f:
	s = line.split()
	if len(s) > 0:
		last = line.split()[-1]
		last_words.append(last)

# Calculate distances of consecutive last words
distances = []
for i in range(len(last_words)-1):
	w1 = last_words[i]
	w2 = last_words[i+1]
	# Distance with reversed word (rhyme ending) and exponential weights
	dist = levenshtein(w1, w2, True, True)
	print(dist)
	distances.append(dist)

distances = np.array(distances)


print("MEAN", np.mean(distances))

# MEAN OF CONSECUTIVE LAST WORDS 0.9876374414207171
# MEAN OF RANDOM WORD PAIRS IN TRAINING DATA 1.4579899821773632