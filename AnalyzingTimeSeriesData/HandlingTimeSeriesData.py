import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading data from input file
def read_data(input_file):
    input_data = np.loadtxt(input_file, delimiter=None)
    dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    output = pd.Series(input_data[:, 2], index=dates)
    return output

if __name__ == '__main__':
    input_file = "/Users/giorgiasavo/Documents/projects/personal/AI_python/AnalyzingTimeSeriesData/timeData.txt"
    timeseries = read_data(input_file)
    plt.figure()
    timeseries.plot()
    plt.show()

    print("\nmean value: ", timeseries.mean())
    print("\nmax value: ", timeseries.max())
    print("\nmin value: ", timeseries.min())
    print("\nstatistic in one time: ", timeseries.describe())

    #resampling with mean
    timeseries_mm = timeseries.resample("A").mean()
    timeseries_mm.plot(style='g--')
    plt.show()

    # resampling with median
    timeseries_mm = timeseries.resample("A").median()
    timeseries_mm.plot()
    plt.show()

    #calculating the moving mean
    timeseries.rolling(window=12, center=False).mean().plot(style='-g')
    plt.show()