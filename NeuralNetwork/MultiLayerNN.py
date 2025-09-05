#a multilayer neural network consists of more than one layer to extract the underlying patterns in the training data

#we are going to generate some data points based on the equation y = 2x^2 + 8

import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

from NeuralNetwork.SingleLayerNN import neural_net

#generates 160 points x equidistant from -30 to 30 and calculates y = 2x^2 + 8
min_val = -30
max_val = 30
num_points = 160
x = np.linspace(min_val, max_val, num_points)
y = 2 * np.square(x) + 8
y /= np.linalg.norm(y) #normalization of y values

#data organized as vectorial columns
data = x.reshape((num_points, 1))
labels = y.reshape((num_points, 1))
#plot the input data to understand how they are distributed
plt.figure()
plt.scatter(data, labels) #shows the point of the original parable
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.title('Input data')
plt.show()

#buiding the neural network having two hidden layers with neurolab with ten neurons in the first hidden layer, six in the second hiden layer and one in the output layer
neural_net = nl.net.newff([[min_val, max_val]], [10, 6, 1]) #creates a feedforward multilayered network
neural_net.trainf = nl.train.train_gd
error = neural_net.train(data, labels, epochs=1000, show=100, goal=0.01) #we are training the network to transform an input x in an output y (so to reconstruct parable)
output = neural_net.sim(data)
y_pred = output.reshape(num_points) #predicted y
plt.figure()
plt.plot(error)
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('Training error progress')
plt.grid()
plt.show()

#plotting acutal and predicted output
x_dense = np.linspace(min_val, max_val, num_points*2)
y_dense_pred=neural_net.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')
plt.show()

#The network starts with random weights: predictions are basically random
#forward pass: Input x is passed through the network.Each neuron computes a weighted sum + activation function. The network outputs a prediction ŷ.
#error: Compare predicted output ŷ with the true output y.
#backward pass: Error is propagated backward through the network to adjust weight
#steps are repeated for all data points and is repeated for many epochs untile the error is smart enough (goal = 0.01) and the maximum number of epochs (1000) is reached


