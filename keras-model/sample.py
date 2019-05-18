# Load LSTM network and generate text
import sys
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

X = np.load("keras-model/X.npy")
Y = np.load("keras-model/Y.npy")
y = np_utils.to_categorical(Y)
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-16-4.2270.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = np.random.randint(0, len(X)-1)
pattern = [0,1,2,5,3,0,1,2,5,3,0,1,2,5,3,0,1,2,5,3]
# generate characters

U = np.load("keras-model/U.npy")
len_U = len(list(U))

runo = ""
for i in range(1000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	#x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	prediction = prediction / np.sum(prediction)

	topN = prediction.flatten()
	topN.sort()
	topN = topN[-100]
	#print(topN)

	numbers = []
	probs = []
	for i in range(len_U):
		#print(prediction[0][i])
		if prediction[0][i] > topN:
			numbers.append(i)
			probs.append(prediction[0][i])

	probs = np.array(probs) / np.sum(np.array(probs))
	#print(np.sum(probs))
	index = np.random.choice(numbers, 1, p=probs)[0]
	#index = np.argmax(prediction)

	word = U[index]
	if word == "RIVINVAIHTO":
		runo += "\n"
	else:
		runo += word + " "
	#result = int_to_char[index]
	#seq_in = [int_to_char[value] for value in pattern]
	#sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print(runo)
