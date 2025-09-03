#also called hierarchical clustering or mean shift cluster analysis
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#generating two-dimensional dataset, containing four blobs
from sklearn.datasets import make_blobs
centers = [[2,2], [4, 5], [3, 10]]
x, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)
plt.scatter(x[:, 0], x[:, 1])
plt.show() #visualizing the dataset

#train the mean shift
ms = MeanShift()
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

#printing the cluster centers and the expected numner of cluster as per the input data
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)

colors = 10*['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 10)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", color='k', s=150, linewidths=5, zorder=10)
plt.show()