#frequently used for prediction of prices, economics, variations and so on
# in regression the relationship between input and output variables mattersa and it helps us in understanding how the value
#of the output variable changes with the change of input variable

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

#providing input data
input = '/Users/giorgiasavo/Documents/projects/personal/AI_python/Regression/linear.txt'
input_data = np.loadtxt(input, delimiter=',')
x, y = input_data[:, :-1], input_data[:, -1]

#training the model with training and testing samples
training_samples = int(0.6*len(x)) #60% of training
testing_samples = len(x) - training_samples
x_train, y_train = x[:training_samples], y[:training_samples] #x_train contiene i primi training_samples campioni.
x_test, y_test = x[training_samples:], y[training_samples:]

#linear regressor object
reg_linear = linear_model.LinearRegression()
reg_linear.fit(x_train, y_train) #training the object with training samples
y_test_pred = reg_linear.predict(x_test) #prediction with the testing data

#plotting and visualizing data
plt.scatter(x_test, y_test, color='red') #real value of the dataset used to test
plt.plot(x_test, y_test_pred, color='black', linewidth=2) #it is the line of regression found by the model, so the prevision of y = f(x)
# if red spots are near the black line it means that the model succedes in explaining well data, if they are scattered it means that the linear model is not so suited
plt.xticks(())
plt.yticks(())
plt.show()

print("performance of linear regressor:")
print("Mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) #Measures the average absolute difference between the true values and the predictions.
print("Mean squared error = ", round(sm.mean_squared_error(y_test, y_test_pred), 2)) #Measures the average of squared differences between actual and predicted values. --> Squaring penalizes big errors more heavily than small ones.
print("Median absolute error = ", round(sm.median_absolute_error(y_test, y_test_pred), 2)) #it takes the median of absolute errors.
print("Exolain variance score = ", round(sm.explained_variance_score(y_test, y_test_pred), 2)) #Measures how much of the variance in the target variable is explained by the model.
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))