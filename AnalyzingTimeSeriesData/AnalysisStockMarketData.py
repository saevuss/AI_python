#going to analyze the data of stock market to get an idea about how the HMM
# works with sequential or time series data

#HMM = Hidden Markov Model is a stochastic model built upon the concept of Markov
# chain based on the assumption that probability of future stats depends only on
# the current process state rather any state that preceded it

import datetime
import warnings
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

import datetime
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# load data from start date and end date
start_date = "1995-10-10"
end_date = "2015-04-25"

# scarica i dati storici di INTC da Yahoo Finance
quotes = yf.download("INTC", start=start_date, end=end_date)

#extracting closing quotes everyday
#closing_quotes = np.array([quote[2] for quote in quotes])
#volumes = np.array([quote[5] for quote in quotes])[1:]
# extracting closing quotes and volumes
closing_quotes = quotes['Close'].values
volumes = quotes['Volume'].values[1:]

#percentage difference of closing stock prices
diff_percentages = 100 * np.diff(closing_quotes) / closing_quotes[:-1]
#training_data = np.column_stack([diff_percentages, volumes])
training_data = np.column_stack([diff_percentages, volumes])
dates = quotes.index[1:].to_numpy()

#creating and training the gaussian HMM
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hmm.fit(training_data)

#generating data using HMM model
num_samples = 300
samples, _ = hmm.sample(num_samples)

plt.figure()
plt.title("Volume of shares")
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)
plt.show()