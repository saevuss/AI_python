''' Convolutional Neural Networks (CNN) solve an image classification
problem that is to which class the input image belongs to

The training archive contains 25,000 images of dogs and cats.'''
import numpy as np
from PIL import ImageFile
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D #to perform pooling operation step #2
from keras.layers import Flatten #process of converting all the resultant 2D arrays into a single long continuous linear vector
from keras.layers import Dense #to perform the full connection of the neural network
from keras.src.legacy.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

S_classifier = Sequential()
S_classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu')) #relu is the rectifier function
#pooling operation on the resultant feature maps after convolution part
S_classifier.add(MaxPooling2D(pool_size=(2, 2)))
#converting all the pooled images ino a continuous vector by using glattering
S_classifier.add(Flatten())
#creating a fully connected layer
S_classifier.add(Dense(units=128, activation='relu')) #128 is the numner of the hidden units
S_classifier.add(Dense(units=1, activation='sigmoid'))

S_classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
#optimizer  parameter  is  to  choose  the  stochastic  gradient  descent
# algorithm,  loss parameter  is  to  choose  the  loss  function
# and  metrics  parameter  is  to  choose  the performance metric.
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("/Users/giorgiasavo/Documents/projects/personal/AI_python/DeepLearning/archive/train_set", target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory("/Users/giorgiasavo/Documents/projects/personal/AI_python/DeepLearning/archive/test_set", target_size=(64, 64), batch_size=32, class_mode='binary')

S_classifier.fit(training_set,steps_per_epoch = 8000,epochs = 25,validation_data = test_set,validation_steps = 2000)
from keras.preprocessing import image
test_image = image.load_img('/Users/giorgiasavo/Documents/projects/personal/AI_python/DeepLearning/archive/single_prediction/947.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = S_classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)








