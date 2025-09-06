''' Learning how to build a linear regressor using artificial neural networks.'''
#impara a predire prezzi delle case utilizzando il dataset boston housing

import numpy
import pandas
from keras import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#loading the dataset which is saved in local directory
dataframe = pandas.read_csv("/Users/giorgiasavo/Documents/projects/personal/AI_python/DeepLearning/Boston.csv", header=None)
'''  12 features (caratteristiche delle case): criminalità, zone industriali, numero stanze, età edifici, ecc.
     1 target (MEDV): prezzo mediano delle case in migliaia di dollari'''
dataset = dataframe.values
x = dataset[:, 0:12]
y = dataset[:, 12]

#defining the model of baseline neural networks
def baseline_model():
    model_regressor = Sequential()
    model_regressor.add(Dense(13, input_dim=12, kernel_initializer='normal', activation='relu')) #input 12 neuroni (una per ogni caratteristica)
    model_regressor.add(Dense(1, kernel_initializer='normal')) #output un neurone (prezzo predetto)

    #compiling the model
    model_regressor.compile(loss='mean_squared_error', optimizer='adam')
    return model_regressor

seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed, shuffle=True) #divisione dei dati in 10 parti (9 per training e una per test=
baseline_result  = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (baseline_result.mean(),baseline_result.std()))
