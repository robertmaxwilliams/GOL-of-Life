import numpy as np

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

def single_dataset(board_dims, number_timelines, timeline_length):
	return cascaded_dataset(board_dims, 0, number_timelines, timeline_length)

def cascaded_dataset(board_dims, cascade_depth, number_timelines, timeline_length, encoder=None):
	"""
	Creates two arrays of training data as such:
	X: a_1,   a_2,   a_3,   a_n+1 ...
	Y: a_1+c, a_2+c, a_3+c, a_n+c ...
	where c is 'cascade depth'
	If encoder is specified, its prediction (the abstracted board)
	is saved instead.
	"""
	X, Y = list(), list()
	for i in range(number_timelines):
		board = np.random.randint(2, size=board_dims)
		for j in range(timeline_length):
			if j + cascade_depth < timeline_length:
				X.append(np.expand_dims(board.copy(), axis=3))
			if j - cascade_depth >= 0:
				Y.append(X[j-cascade_depth])	
			iterate(board)
	X = np.array(X)
	Y = np.array(Y)
	if encoder != None:
		X = encoder.predict(X)
		Y = encoder.predict(Y)

	return X, Y


