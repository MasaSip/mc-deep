f = open("whole_dataset.txt", "r").read().lower()

words = {}
for w in f.split():
	if w not in words:
		words[w] = 1
	else:
		words[w] = words[w] + 1


numbers = []
for key, value in words.items():
	if value < 200:
		numbers.append(value)

print(numbers)

import matplotlib.pyplot as plt

plt.hist(numbers, bins=200)
plt.yscale('log')
plt.show()