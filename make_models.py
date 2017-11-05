"""
Makes all models needed for gol simulator
autoencoder: learns a reduced model for GOL board
stepper: calculates the next step given a reduced board model

(board) -> encode -> (abstracted board) -> decoder -> (board)
(abstracted board) -> stepper -> (next abstracted board)

What is the best way to calculate gradient fot the abstracted board?
Should is be compared to the encoded iterate(board) or should it be decoded and gradients backpropagated from the iterate(board)? Are there processes equivalent? What loss function is best for each case?
"""
import keras
import os
import numpy as np
import gol_datagen
import gol_graphics

def create_autoencoder():
	"""
	creates convolutional autoencoder for use with
	gol grids. create compressed abstraction with
	spacial relations etc included
	"""

	# create model
	encoder_input = keras.layers.Input(shape=(None, None, 1))
	encoder_output = keras.layers.Conv2D(20, (7,7), strides=(3,3), padding='same', activation='relu')(encoder_input)
	encoder_output = keras.layers.Conv2D(20, (7,7), strides=(1,1), padding='same', activation='relu')(encoder_output)


	# create input tensor	
	decoder_input = keras.layers.Input(shape=(None,None,20))

	# inner decoding
	decoder_output = keras.layers.Deconv2D(20, (7,7), strides=(3,3), padding='same', activation='relu')(decoder_input)
	decoder_output = keras.layers.Deconv2D(1, (7,7), strides=(1,1), padding='same', activation='sigmoid')(decoder_output)

	# outer decoding
	#decoder_output = keras.layers.Deconv2D(20, (5,5), strides=(1,1), padding ='same', activation='sigmoid')(x)
	

	# create our three models
	encoder = keras.models.Model(encoder_input, encoder_output)
	decoder = keras.models.Model(decoder_input, decoder_output)
	autoencoder = keras.models.Model(encoder.input, decoder(encoder.output))

	# compile those models
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	decoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

	# save to file and return for immediate use
	# TODO is this needed? Model creation is very fast, and interdependencies
	# are not preserved, ie training autoencoder no longer affects encdoer and decoder
	#autencoder.save(model_filename)
	#encoder_model.save('encoding_' + model_filename)
	#decoder_model.save('decoding_' + model_filename)

	return autoencoder, encoder, decoder

def save_autoencoder(encoder, decoder, base_filename):
	encoder.save('models/' + base_filename + '_encoder.h5')
	decoder.save('models/' + base_filename + '_decoder.h5')
def load_autoencoder(base_filename):
	encoder = keras.models.load_model('models/' + base_filename + '_encoder.h5')
	decoder = keras.models.load_model('models/' + base_filename + '_decoder.h5')
	autoencoder = keras.models.Model(encoder.input, decoder(encoder.output))


	# compile those models
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	decoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

	print(autoencoder.summary())
	return autoencoder, encoder, decoder

def train_autoencoder(autoencoder, epochs=5, board_dims=(81,81), number_timelines=10, timeline_length=50, debug=False): 
	""" 
	Trains autoencoder model, as above, using gol_datagen, as above.
	"""

	X, Y = gol_datagen.single_dataset(board_dims, number_timelines, timeline_length)	
	if debug:
		print(X[1], Y[1])
	print(X.shape, "X shape")
	#note that input = output, we discard Y.
	print(autoencoder.summary())
	autoencoder.fit(X, X, epochs=epochs, batch_size=5, shuffle=True)

def create_stepper():
	"""
	take 2d grid of abstracted GOL board and predicts the next n steps
	TODO figure out attention and variable computation time
	TODO compare repeated feedforward vs recurrent (will state help?)
	"""

	#create model
	stepper_input = keras.layers.Input(shape=(None, None, 20))

	# dense layers
	stepper_output = (keras.layers.core.Dense(64, activation='relu'))(stepper_input)
	stepper_output = (keras.layers.core.Dense(64, activation='relu'))(stepper_input)
	stepper_output = (keras.layers.core.Dense(20, activation='relu'))(stepper_input)

	stepper = keras.models.Model(stepper_input, stepper_output)

	# complile
	stepper.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy', 'mse'])

	return stepper

def train_stepper(stepper, encoder, decoder, epochs): 
	"""
	Uses the top half of the autoencoder, the encoder, 
	to create abstracted boards and the abstraction of the board 1 step later
	"""

	X, Y = gol_datagen.cascaded_dataset((81,81), 1, 10, 100, encoder=encoder)
	print(X.shape, "X shape")
	print(stepper.summary())
	
	stepper.fit(X, Y, epochs=epochs, batch_size=5, shuffle=True)

def save_stepper(stepper, base_filename):
	stepper.save('models/' + base_filename + '_stepper.h5')
def load_stepper(base_filename):
	stepper = keras.models.load_model('models/' + base_filename + '_encoder.h5')

	print(stepper.summary())
	return stepper


filename = 'two_layer_81_50'
generate = False
if generate:
	autoencoder, encoder, decoder = create_autoencoder()
	train_autoencoder(autoencoder, epochs=10, board_dims=(81,81))
	save_autoencoder(encoder, decoder, filename)
else:
	autoencoder, encoder, decoder = load_autoencoder(filename)
print("made autoencoder and friends")


generate_s = False

if generate_s:
	stepper = create_stepper()
	train_stepper(stepper, encoder, decoder, epochs = 500)
	save_stepper(stepper, filename)
else:
	stepper = load_stepper(filename)

X, Y = gol_datagen.single_dataset((81,81), 10, 100)
outs = encoder.predict(X)
gol_graphics.animate((outs[j,:,:,j] for j in range(100)), delay_time=0.2)
"""
X, Y = gol_datagen.single_dataset((81,81), 10, 100)
def discretize(cell):
	if cell >= 0.5:
		return 1
	else:
		return 0
discretize = np.vectorize(discretize)


def demo():
	for i in range(200):
		prediction = np.squeeze(autoencoder.predict(X[i:i+1]))	
		discrete = discretize(prediction.copy())
		truth = np.squeeze(X[i])
		boarder = np.ones((3, truth.shape[1]))
		yield np.concatenate((prediction, boarder, discrete, boarder, truth), axis=0)
	

gol_graphics.animate(demo(), window_shape=(900,300))
"""

