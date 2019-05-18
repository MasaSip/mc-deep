import re
import numpy as np

text = open('demos/embedding_demo/heikkikuula.txt').read()

#text = re.sub(r'[^äö]', r'', text)
text = re.sub(r"[^a-zA-Z \n]", r"", text)
text = re.sub(r"\n", r" RIVINVAIHTO ", text)
text = text.split(' ')
unique_word = list(set(text))


