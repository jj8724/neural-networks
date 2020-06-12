import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# define nonlinear dataset inputs - the 2 moons
np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
# plt.show()

# each point is now its own matrix consisting of 1 row
labels = labels.reshape(100,1)

# define the sigmoid function (taken from the sigmoidtest.py example)
def sigmoid(x):
	return 1/(1+np.exp(-x))

# derivative of sigmoid function
def sigmoid_der(x):
	return sigmoid(x)*(1-sigmoid(x))
	
# set random hidden weights, the number of which is set by the number of feature sets
# and the number of weights per feature set
wh = np.random.rand(len(feature_set[0]), 4)

# set random output weights, similar to previous perceptron this is governed by the number of
# 'inputs' to this layer which is 4 
wo = np.random.rand(4,1)

# set the learning rate
lr = 0.5

# start the iteration loop by defining the number of iterations
for epoch in range(2000):
	
	# FEEDFORDWARD
	#--------------
	# start by defining the feature set as the hidden layer inputs
	inputs = feature_set
	
	# the sum of the hidden layer inputs and corresponding weights
	zh = np.dot(inputs, wh)
	
	# sigmoid of that to get value between 0 and 1
	ah = sigmoid(zh)
	
	# the sum of the hidden layer inputs and corresponding output weights
	zo = np.dot(ah, wo)
	
	# sigmoid of output to get value between 0 and 1
	ao = sigmoid(zo)
	
	# BACKPROPAGATION
	# -----------------
	# calculate the MSE, note that n=2 for the number of coordinates for each point
	error_out = (1/2) * (np.power((ao - labels), 2))
	# print(error_out.sum())
	
	# calculate cost function for the output layer using chain rule i.e. dJ/dwo = dJ/dao * dao/dzo * dzo/dwo
	dJ_dao = ao - labels
	dao_dzo = sigmoid_der(zo)
	dzo_dwo = ah
	
	# Transpose due to matrix multiplication operations - don't understand, need some help
	dJ_dwo = np.dot(dzo_dwo.T, dJ_dao * dao_dzo)
	#
	# But that only gives output layer weights, we still need to calculate the hidden layer weights
	#
	# Same partial differential eqns to minimise the cost function of hidden layer output
	# dJ/dwh = dJ/dah * dah/dzh * dzh/dwh
	# but we don't know dJ/dah so we must use the chain rule again there
	# dJ/dah = dJ/dzo * dzo/dah
	# but we do't know dJ/dzo so we need to use the chain rule again
	# dJ/dzo = dJ/dao * dao/dzo
	# When combined this gives: dJ/dwh = dJ/dao * dao/dzo * dzo/dah * dah/dzh * dzh/dwh
	dJ_dzo = dJ_dao * dao_dzo
	dzo_dah = wo
	dJ_dah = np.dot(dJ_dzo, dzo_dah.T)
	dah_dzh = sigmoid_der(zh)
	dzh_dwh = inputs
	dJ_dwh = np.dot(dzh_dwh.T, dah_dzh * dJ_dah)
	
	wh -= lr * dJ_dwh
	wo -= lr * dJ_dwo

a = float(input('Input x coordinate'))
b = float(input('Input y coordinate'))

single_point = np.array([a, b])
hidden_result = sigmoid(np.dot(single_point, wh))
result = sigmoid(np.dot(hidden_result, wo))
print(result)

if result >= 0.75:
	print("Your point is highly likely to be in the green dataset!")
elif result >= 0.5 and result < 0.75:
	print("Your point is probably but not definitely in the green dataset!")
elif result > 0.25 and result < 0.5:
	print("Your point is probably but not definitely in the blue dataset!")
elif result <= 0.25:
	print("Your point is highly likely to be in the blue dataset!")
