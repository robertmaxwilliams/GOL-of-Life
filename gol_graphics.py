import pygame
import numpy as np
import time


def draw(value_array, window_shape=(500,500)):
	"""
	takes a 2d numpy array and draws it on a window untill x button is pressed
	"""
	pygame.init()
	display = pygame.display.set_mode((350, 350))
	value_array = 255*value_array/value_array.max()
	surf = pygame.surfarray.make_surface(value_array)
	surf = pygame.transform.scale(surf, window_shape)
	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
		display.blit(surf, (0, 0))
		pygame.display.update()
	pygame.quit()

def animate(value_array_generator, max_val=0, delay_time=0.2, window_shape = (500,500)):
	pygame.init()
	display = pygame.display.set_mode(window_shape)
	running = True
	wait_s = delay_time 
	time_start = 0 
	while running: 
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
		if (time.time() - time_start > wait_s):
			try:
				value_array = value_array_generator.next()
			except StopIteration:
				running = False
				break
			if max_val != 0:
				value_array = 255*value_array/max_val
			elif value_array.max() != 0:
				value_array = 255*value_array/value_array.max()
			surf = pygame.surfarray.make_surface(value_array)
			surf = pygame.transform.scale(surf, window_shape)
			display.blit(surf, (0, 0))
			time_start = time.time()
		pygame.display.update()
	pygame.quit()

def test_iteration_model(model_filename):
	import keras
	autoencoder = keras.models.load_model(model_filename)


	def generate_boards(model, board_dims):
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
			border = np.ones((3, board_dims[1]))
			out_image = np.reshape(board_sim, board_dims)
			out_image = np.append(out_image, border, axis=0)
			yield np.append(out_image, 2*board_real, axis=0)
							
	animate(generate_boards(autoencoder, (81*3,81*3)), max_val=1, delay_time=0.0, window_shape=(1000, 500))


#animate((np.random.randint(0, high=2, size=(100, 10*x)) for x in range(1, 20)))
