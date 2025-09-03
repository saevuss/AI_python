import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

#two-dimensional dataset, containing four blobs
from sklearn.datasets._samples_generator import make_blobs
x, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()



