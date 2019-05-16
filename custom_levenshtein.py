import numpy as np


# n is the maximum length of a word
n = 100
invx = np.array(range(1,n+1))
weights_pol = np.ones(n) / (invx ** 1.5)
weights_exp = np.exp(-(invx - 1))

# if reverse, letters in the beginning of a word have more weight
# otherwise, letters in the end of a word have more weight
# the weight decays with the function 1/x^1.5 of indices 
# counted from the end/beginning

def levenshtein(s1, s2, reverse=False, exp=False):
	if exp:
		weights = weights_exp
	else:
		weights = weights_pol

	if reverse:
		s1 = s1[::-1]
		s2 = s2[::-1]

	if len(s1) < len(s2):
		return custom_levenshtein(s2, s1, weights)
	elif len(s1) == len(s2):
		d1 = custom_levenshtein(s1, s2, weights)
		d2 = custom_levenshtein(s2, s1, weights)
		return min(d1, d2)
	else:
		return custom_levenshtein(s1, s2, weights)


def custom_levenshtein(s1, s2, weights):
	n_1 = len(s1)
	#print("PR1", previous_row)
	previous_row = [0.0] + list(weights[:(n_1+1)])

	for i in range(0, n_1+1):
		previous_row[i + 1] = previous_row[i] + previous_row[i + 1]

	previous_row = previous_row[:(n_1+2)]
	first_row = list(previous_row)

	for i, c1 in enumerate(s1):

		current_row = [first_row[i+1]]
		for j, c2 in enumerate(s2):
			weight_i = weights[j+1]
			insertions = previous_row[j + 1] + weight_i
			deletions = current_row[j] + weights[j]
			substitutions = previous_row[j] + (c1 != c2) * weights[j]
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row
	return previous_row[-1]
