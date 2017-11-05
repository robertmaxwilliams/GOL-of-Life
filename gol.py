from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Deconv2D
from keras.optimizers import RMSprop
from keras import losses, layers
from keras.models import load_model

import os
import gol_graphics
from gol_iterate import iterate
from multiprocessing import Process
board = np.random.randint(2, size=(10, 10))

def generate_data():
	global board
	board = iterate(board)
	return board

def update(data):
    mat.set_data(data)
    return mat 

def data_gen():
    while True:
        yield generate_data()

#now the fun part: take board_n as input, and board_n+1 as output.
# generate random board, zag back and forth 10 iterations,
# then make a new board
def generate_data(board_dims):
	X, y = list(), list()
	for n in range(10):
		board = np.random.randint(2, size=board_dims)
		for _ in range(100):
			X.append(board.copy())
			iterate(board)
			y.append(board.copy())
	X = np.array(X)
	y = np.array(y)
	X = np.expand_dims(X, 3)
	y = np.expand_dims(y, 3)
	return X, y

model_name = 'models/gol_net9.h6'
overwrite = False 
training_board_shape = (3**4,3**4)
if os.path.exists(model_name) and not overwrite:
	model = load_model(model_name)
else:
	X, y = generate_data(training_board_shape)
	model = Sequential()
	model.add(Conv2D(6, (5,5), input_shape=(None, None, 1), strides=(1, 1), padding='same'))
	model.add(Activation('relu'))
	# reducing convolution
	model.add(Conv2D(6, (7,7), strides=(3, 3), padding='same'))
	model.add(Activation('relu'))
	# dense
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	# deconvolution
	model.add(Deconv2D(6, (7, 7), strides=(3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Conv2D(1, (5,5), strides=(1, 1), padding='same'))

	model.compile(loss='mean_squared_error',
				  optimizer='adam',
				  metrics=['accuracy', 'mse'])


	model.fit(X, y, epochs=10, batch_size=5)
	model.save(model_name)
		
board_dims = (3**5, 3**5)
def generate_boards():
	board_sim = np.random.randint(2, size=board_dims)
	board_sim = iterate(board_sim)
	board_real = board_sim.copy()
	board_sim = np.expand_dims(board_sim, 0)
	board_sim = np.expand_dims(board_sim, 3)
	for _ in range(1000):
		board_sim = model.predict(board_sim)
		discretize = np.vectorize(lambda x: 0 if x<0.5 else 1)
		board_sim = discretize(board_sim)
		board_sim = board_sim.astype(int) 
		board_real = iterate(board_real)
		image = np.concatenate((board_real, np.ones((3, 3**5))), axis=0) image = np.concatenate((image, np.reshape(board_sim, board_dims)), axis=0)
		yield image
print(model.summary())
#gol_graphics.animate(generate_boards(), max_val=3, delay_time=0.0)

def generate_train_boards():
	X, y = generate_data((20,20))
	for board_now, board_next in zip(X, y):
		board_now = np.reshape(board_now, (20, 20))
		board_next = np.reshape(board_next, (20, 20))
		yield board_now*2 + board_next
gol_graphics.animate(generate_boards(), max_val=5, delay_time=0.0, window_shape=(1000,500))
