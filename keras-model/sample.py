# Load LSTM network and generate text
import sys
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint

X = np.load("keras-model/X.npy")[:,:,0]
Y = np.load("keras-model/Y.npy")
y = np_utils.to_categorical(Y)

linebreak_vec = np.zeros((23991,1))
linebreak_vec[127] = 1.0

e_beginning = np.load("data/embeddings/embedding-beginning.npy")
e_end = np.load("data/embeddings/embedding-end.npy")[:,:63]
embedding_matrix = np.concatenate((e_beginning, e_end), axis=1)
embedding_matrix = np.concatenate((embedding_matrix, np.zeros((1,127))), axis=0)
embedding_matrix = np.concatenate((embedding_matrix, linebreak_vec), axis=1)
vocab_size = len(embedding_matrix)

print(embedding_matrix.shape, vocab_size)

e = Embedding(vocab_size, 128, weights=[embedding_matrix], input_length=20, trainable=False)


print(X.shape,Y.shape)
# define the LSTM model
model = Sequential()
model.add(e)
model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(128, activation='linear'))

d = Dense(y.shape[1], activation='softmax', trainable=False)
model.add(d)

print(d.get_config())
print(d.get_weights())
d.set_weights([embedding_matrix.T, np.zeros(23991)])
# log-loss
# load the network weights
print("load weights")
filename = "data/keras-checkpoints/weights-improvement-03-8.0148.hdf5"
model.load_weights(filename)
print("Compile model...")
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("DONE!")
# pick a random seed
start = np.random.randint(0, len(X)-1)
pattern = list(np.random.randint(0,23991, 20))
#pattern = [0,1,2,5,3,0,1,2,5,3,0,1,2,5,3,0,1,2,5,3]
# generate characters

U = np.load("keras-model/U.npy")
len_U = len(list(U))
print(U.shape)

runo = ""
for i in range(400):
	x = np.reshape(pattern, (1, len(pattern)))
	#x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	prediction = prediction / np.sum(prediction)
	print(prediction)
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
