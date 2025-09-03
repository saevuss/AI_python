import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

#two-dimensional dataset, containing four blobs
from sklearn.datasets._samples_generator import make_blobs
x, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)

kmeans = KMeans(n_clusters=4)
#training the k-means model with the input data
kmeans.fit(x)
y_kmeans = kmeans.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
