import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras
from math import log

prefix = ""#'../'
toydata = False
if toydata:
	X = np.load(prefix + "keras-model/X_0.npy")[:500,:]
	Y = np.load(prefix + "keras-model/Y_0.npy")[:500,:]
else:
	X = np.load(prefix + "keras-model/X.npy")[:,:,0]
	Y = np.load(prefix + "keras-model/Y.npy")

#log(0)
print("To categorical")
#y = np_utils.to_categorical(Y)
print("done")
#print(X.shape, Y.shape, y.shape)

print(Y)

linebreak_vec = np.zeros((23991,1))
linebreak_vec[127] = 1.0

e_beginning = np.load(prefix + "data/embeddings/embedding-beginning.npy")
e_end = np.load(prefix + "data/embeddings/embedding-end.npy")[:,:63]

embedding_matrix = np.concatenate((e_beginning, e_end), axis=1)
embedding_matrix = np.concatenate((embedding_matrix, np.zeros((1,127))), axis=0)
embedding_matrix = np.concatenate((embedding_matrix, linebreak_vec), axis=1)
vocab_size = len(embedding_matrix)

print(embedding_matrix.shape, vocab_size)


rand_ematr = np.random.rand(embedding_matrix.shape[0], embedding_matrix.shape[1])-0.5

e = Embedding(vocab_size, 128, weights=[rand_ematr], input_length=20, trainable=False)


print(X.shape,Y.shape)
# define the LSTM model
model = Sequential()
model.add(e)
model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(128, activation='linear'))

print(23991)
d = Dense(23991, activation='softmax', trainable=False)
model.add(d)

print(d.get_config())
print(d.get_weights())
U_log = np.load(prefix + "keras-model/U_log.npy")

# log-loss
#filename = "data/keras-checkpoints/naive-weights-06-6.6985.hdf5"
#model.load_weights(filename)
d.set_weights([rand_ematr.T, U_log * 1.0])


model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = prefix + "data/keras-checkpoints/naive-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)

