dataset = "whole_dataset.txt"

f = open(dataset, "r").readlines()

def cleanUp(s):
	s = s.replace(".", " .")
	s = s.replace(",", " ,")
	s = s.replace("?", " ?")
	s = s.replace("!", " !")
	s = s.replace("' ", " ")
	s = s.replace("\"", "")
	s = s.lower()
	return s

f_wordlist = []
for l in f:
	l = cleanUp(l).split()

	f_wordlist = f_wordlist + l
	f_wordlist.append("RIVINVAIHTO")

input_len = 10

f = open("formatted_" + dataset, "w")

for ix, w in enumerate(f_wordlist):
	print(ix)
	if ix + input_len < len(f_wordlist):
		sentence = f_wordlist[ix: ix + input_len]
		row = ' '.join(sentence[:-1])
		row = sentence[-1] + "\t"+ row+ "\n"
		f.write(row)


f.close()