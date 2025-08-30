import numpy as np
from sklearn import preprocessing

input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])

#************* DATA PROCESSING **************
#BINARIZATION
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data) #all values above threshold would be converted to 1, and all the values below 0.5 to 0
print("\nBinarized data:\n", data_binarized)

#MEAN REMOVAL
#used to elimiante the mean from feature vector so that every feature is centred on zero
print("Mean =", input_data.mean(axis=0)) #input data is in 2D so axis = 0 is row and axis = 1 is column, so that axis = 0 means that the result is a row with the operation on each column of the first row
print("Standard deviation =", input_data.std(axis=0)) #indica quanto i valori di un insieme di dati sono dispersi o raggruppati attorno alla loro media aritmetica
data_scaled = preprocessing.scale(input_data) #standardizza colonna per colonna, ovvero sottrae la media della colonna e divide per la sua deviazione standard.
print("Mean =", data_scaled.mean(axis=0))
print("Standard deviation =", data_scaled.std(axis=0))

#SCALING
#transoforms data so that every column has values between 0 and 1
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) #it calculates min and max values of each column, min values of each colum becomes 0, the greatest one 1
print("\nMin-max scaled data:\n", data_scaled_minmax)

#NORMALIZATION L1
#least absolute deviations. The values are modified so that the sum of the absolute values is always up to 1 in each row
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("\nNormalized L1 data:\n", data_normalized_l1) # for each row is calculated the norm |n| = |x1|+|x2|+|x3| and each element of the row is divided by this sum

#NORMALIZATION L2
#least squares. modifies the values so that the sum of the squares is always up to 1 in each row -> |n| = √( x1^2 + x2^2 + x3^2 )
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nNormalized L2 data:\n", data_normalized_l2)


#************* LABELLING **************
#sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

#creating the label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#encoding a set of labels
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels) #firstly the string are reordered alphabetically, then it is assigned the number
print("\nLabels =", test_labels)
print("\nEncoded values =", list(encoded_values))

#decoding a set of values
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded values =", list(decoded_list))











