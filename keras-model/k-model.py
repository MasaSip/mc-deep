# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

prefix = '../'

X = np.load(prefix + "keras-model/X.npy")
Y = np.load(prefix + "keras-model/Y.npy")
y = np_utils.to_categorical(Y)

e_beginning = np.load(prefix + "data/embeddings/embedding-beginning.npy")
e_end = np.load(prefix + "data/embeddings/embedding-end.npy")
embedding_matrix = np.concatenate((e_beginning, e_end), axis=1)
vocab_size = len(embedding_matrix)

print(embedding_matrix.shape, vocab_size)

e = Embedding(vocab_size, 128, weights=[embedding_matrix], input_length=20, trainable=False)


print(X.shape,Y.shape)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# log-loss
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="data/keras-checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)