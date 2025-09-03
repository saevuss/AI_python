#in order to build recommender systems (such as movie recommender) we need to understand
#the concept of finding the nearest neighbors -> finding the closest point to the input point form the given dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

#input data
x = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9],[8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9],])

#defining the nearest neighbors
k=3

#giving thest data from which the nearest neighbors is to be found
test_data = [3.3, 2.9]
plt.figure()
plt.title("input data")
plt.scatter(x[:, 0], x[:, 1], marker="o", color='black', s=100)
plt.show()

#building the k nearest neighbor
knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
distances, indices = knn_model.kneighbors([test_data])
print("\nK nearest neighbors:")
#printing the k nearest neighbors
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank)+" is", x[index])

#visualizing the nearest neighbors along with the test data point
plt.figure()
plt.title('Nearest Neighbors')
plt.scatter(x[:, 0], x[:, 1], marker="o", color='k', s=100)
plt.scatter(x[indices][0][:][:, 0], x[indices][0][:][:, 1], marker="o", color='k', s=250, facecolors='none')
plt.scatter(test_data[0], test_data[1], marker="x", s=100, color="k")
plt.show()
