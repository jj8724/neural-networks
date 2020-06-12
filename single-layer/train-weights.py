import numpy as np
import matplotlib.pyplot as plt

# define original dataset inputs
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])

# define answers we are trying to predict, known as labels
labels = np.array([[1,0,0,1,1]])

# convert the single array of answers into 5 arrays, each containing a single number
labels = labels.reshape(5,1)

# set the random number seed so that the random numbers are the same every time it is run
np.random.seed(42)

# set random initial weights
weights = np.random.rand(3,1)

# set a random initial bias
bias = np.random.rand(1)

# set the learning rate
lr = 0.05

# define the sigmoid function (taken from the sigmoidtest.py example
def sigmoid(x):
	return 1/(1+np.exp(-x))

# derivative of sigmoid function
def sigmoid_der(x):
	return sigmoid(x)*(1-sigmoid(x))

# now to train the neural net

# first we define the number of iterations
for epoch in range(20000):
	
	# define the feature_set as input variables
	inputs = feature_set
	
	# Step 1a: using randomly assigned weights/bias, calculate the dot product
	XW = np.dot(inputs, weights) + bias
	
	# Step 1b: normalise or 'squash' the value using the sigmoid function
	z = sigmoid(XW)
	
	# Step 2a: calculate the error by subtracting the actual value from the predicted value
	error = z - labels
	
	# sum the errors
	print(error.sum())
	
	# Step 2b: calculate the gradient of the error and minimise
	
	# first we need to get derivative of the cost function using the chain rule i.e. dJ/dw = dJ/dXW * dXW/dz * dz/dw
	dJdXW = error
	dXWdz = sigmoid_der(z)
	
	dJdz = dJdXW * dXWdz
	
	# we the derivative of z wrt weights will just give the correspeonding inputs (x values)
	# so we can transpose the matrix to group them by input i.e. 3 matrices, one for each of x1, x2, x3
	inputs = feature_set.T
	
	# then we calculate the new weight by subtracting the gradient at each w from the current value of x
	# note that if gradient is postive w will decrease, if gradient is negative w will increase
	# note that 'weights -= ' is the same as 'weights = weights - '
	weights -= lr * np.dot(inputs, dJdz)
	
	# for each iteration of the derivative we need to update the bias
	for num in dJdz:
		bias -= lr * num
		
print(weights)
