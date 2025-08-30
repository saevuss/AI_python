import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris() #loading the input data
x = iris.data[:, :2]
y = iris.target #taking the first two features

#we plot the SVM boundaries with original data
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
h = (x_max - x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #creates the grid that covers all the graph's area
x_plot = np.c_[xx.ravel(), yy.ravel()] #transforms the gird in a x,y array to have predictions
c = 1.0
#creating  the SVM classifier object
svc_classifier = svm.SVC(kernel='linear', C=c, decision_function_shape='ovr').fit(x, y) #creates and trains the model SVM
z = svc_classifier.predict(x_plot)
z = z.reshape(xx.shape) #reports the predictions in the original grid
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')

plt.show()


