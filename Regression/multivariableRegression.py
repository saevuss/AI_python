import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

input = '/Users/giorgiasavo/Documents/projects/personal/AI_python/Regression/mul_linear.txt'
input_data = np.loadtxt(input, delimiter=',')
x, y = input_data[:,:-1], input_data[:,-1]
training_samples = int(0.6*len(x))
test_samples = len(x)-training_samples

x_train, y_train = x[:training_samples], y[:training_samples]
x_test, y_test = x[training_samples:], y[training_samples:]
reg_linear_mul = linear_model.LinearRegression()
reg_linear_mul.fit(x_train, y_train)
y_test_pred = reg_linear_mul.predict(x_test)

print("performance of linear regressor:")
print("Mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) #Measures the average absolute difference between the true values and the predictions.
print("Mean squared error = ", round(sm.mean_squared_error(y_test, y_test_pred), 2)) #Measures the average of squared differences between actual and predicted values. --> Squaring penalizes big errors more heavily than small ones.
print("Median absolute error = ", round(sm.median_absolute_error(y_test, y_test_pred), 2)) #it takes the median of absolute errors.
print("Exolain variance score = ", round(sm.explained_variance_score(y_test, y_test_pred), 2)) #Measures how much of the variance in the target variable is explained by the model.
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))

#polynomial of degree 10 to train the regressor
polynomial = PolynomialFeatures(degree = 10)
x_train_transformed = polynomial.fit_transform(x_train)
datapoint = [[2.23]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(x_train_transformed, y_train)
print("\nLinear regression:\n", reg_linear_mul.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))