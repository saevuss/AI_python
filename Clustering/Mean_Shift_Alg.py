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

