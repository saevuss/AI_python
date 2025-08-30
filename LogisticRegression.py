import numpy as np
from fontTools.misc.classifyTools import Classifier
from sklearn import linear_model
import matplotlib.pyplot as plt

def Logistic_visualize(Classifier_LR, x, y):
    min_x, max_x = x[:, 0].min() - 1.0, x[:, 0].max() + 1.0
    min_y, max_y = x[:, 1].min() - 1.0, x[:, 1].max() + 1.0
    mesh_step_size = 0.02
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size)) #creation of a grid
    output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()]) #classifies every points of the grid so that it is simple to know the class of a general point in the grid
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray) #regions are colored

    plt.scatter(x[:, 0], x[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(x[:, 0].min()-1), int(x[:, 0].max()+1), 1.0)))
    plt.yticks((np.arange(int(x[:, 1].min() - 1), int(x[:, 1].max() + 1), 1.0)))
    plt.show()

x = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4], [3.2, 0.9], [2.9, 1.9], [2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]]) #12 points of x, y coords
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]) #12 labelsn every point belongs to one class

#creating logistic regression classifier
Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C=75)
Classifier_LR.fit(x, y) #train the classifier
Logistic_visualize(Classifier_LR, x, y)

