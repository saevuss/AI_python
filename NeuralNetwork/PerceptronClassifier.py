import matplotlib.pyplot as plt
import neurolab as nl

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]
#creating a network with 2 inputs and 1 neuron
net = nl.net.newp([[0, 1], [0, 1]], 1)
#training the network
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1)
#visualizing the output
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()
