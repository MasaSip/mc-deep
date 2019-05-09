import numpy as np

# n is the maximum length of a word
n = 100
invx = np.array(range(1,n+1))
weights = np.ones(n) / (invx ** 1.5)

# if reverse, letters in the beginning of a word have more weight
# otherwise, letters in the end of a word have more weight
# the weight decays with the function 1/x^1.5 of indices 
# counted from the end/beginning

def levenshtein(s1, s2, reverse=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1,reverse=reverse)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    if reverse:
        s1 = s1[::-1]
        s2 = s2[::-1]

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] +weights[j+1]
            # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + weights[j]
            # than s2
            substitutions = previous_row[j] + (c1 != c2) * weights[j]
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

