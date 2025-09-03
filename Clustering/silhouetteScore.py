import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn import metrics

sns.set()
import numpy as np
from sklearn.cluster import KMeans

#two-dimensional dataset
from sklearn.datasets._samples_generator import  make_blobs
x, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=0.40, random_state=0)
#inizialize variable
scores = []
values = np.arange(2, 10)
for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(x)

#estimating the silhouette score for the current clustering model using the euclidean distance metric
score = metrics.silhouette_score(x, kmeans.labels_, metric='euclidean', sample_size=len(x))
print("\n number of clusters=", num_clusters)
print("\n silhouette score=", score)
scores.append(score)

num_clusters = np.argmax(scores) + values[0]
print("\n optimal number of clusters=", num_clusters)