#we are creating a single layer neural network that consists of
#independent neurons acting on input data to produce the output

import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

input_data = np.loadtxt("/Users/giorgiasavo/Documents/projects/personal/AI_python/NeuralNetwork/neural_simple.txt")
#separatiung columns
data = input_data[:, 0:2]
labels = input_data[:, 2:] #target values expected from the net
#plot the input data to understand how they are distributed
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.title('Input data')
plt.show()

#min and max values of both dimensions to normalize net's input
dim1_min, dim1_max = data[:, 0].min(), data[:, 0].max()
dim2_min, dim2_max = data[:, 1].min(), data[:, 1].max()

#defining number of neurons in the output layer
nn_output_layer = labels.shape[1]

#defining a single-layer neural network
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)
#training the nn
error = neural_net.train(data, labels, epochs=200, show=20, lr=0.01)
plt.figure()
plt.plot(error)
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('Training error progress')
plt.grid()
plt.show()

print('\nTest Results:')
data_test = [[1.5, 3.2], [3.6, 1.7], [3.6, 5.7],[1.6, 3.9]]
for item in data_test:
    print(item, '-->', neural_net.sim([item])[0])

