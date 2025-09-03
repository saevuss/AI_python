#KNN = K-Nearest Neighbors
#this classifiere uses the nearest neighbots algorithm to classify a given data point
# the KNN classifiers have a fixed user defined constant for the numner of neighbors which have to be determined
from ast import increment_lineno


from sklearn.datasets import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def image_display(i): #takes image number i
    plt.imshow(digit['images'][i], cmap='Greys_r')
    plt.show()

digit = load_digits() #creates a dataset which contains images 8x8 of written numbers
digit_d = pd.DataFrame(digit['data'][0:1600]) #datafrem with the first 1600 examples
image_display(9)

#training and testing data set
train_x = digit['data'][:1600] #first 1600 examples as features
train_y = digit['target'][:1600] #first 1600 examples as labels
KNN = KNeighborsClassifier(20) #to classificate a new point it looks at the 20 nearest examples
KNN.fit(train_x, train_y) #TRAINING
#creates the k nearest neighbor classifier constructor
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=20, p=2, weights='uniform')
test = np.array(digit['data'][1725]) #takes the image with index 1725 and puts it into numpy array
test1 = test.reshape(1, -1) #transofrm in a format that scikit-learn accepts (a 2d Array)
image_display(1725) #shows the image
print("Prediction:", KNN.predict(test1)) #calculates the distance between the new point and all the point of the training (euclidean) and takes the 20 nearest points
#looks at the labels of the 20 points and assign to the new point the label of the most frequent one among the 20 nearest
print("Real label:", digit['target'][1725])

