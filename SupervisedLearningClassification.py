import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer() #to load dataset
#creating new variables for each important set of information
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#ORGANIZING DATA INTO SETS --> we will divide our data into two parts namely a training
#set and a test set. we have to test our model on the unseen data
#train_test_split() is a function to split the data into sets

train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.40, random_state=42) #splitting into training and test data 40% if the data for testing the other for training

#BUILDING THE MODEL: using Naive Bayes algorithm
gnb = GaussianNB() #inizializing the model
model = gnb.fit(train, train_labels) #train the model by fitting it to the data by using gnb.fit()

#EVALUATING THE MODEL AND ITS ACCURANCY
#evaluating by making predictions on our test data using predict() function
preds = gnb.predict(test)

#by comparing test_labels and preds we can find out the accuracy of our model

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, preds)) #the naive classifier is 95.17% accurate

















