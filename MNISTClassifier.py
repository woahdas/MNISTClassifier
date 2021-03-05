#I tried to not use the Kaggle .csv train data but rather load the data in from Keras, 
#but Kaggle would stop working after getting a connection with the server to download the data (so it wouldn't even download the data, it would just connect?).
#So I made another model and used it in python, but this was the model that I first made in Kaggle. The working model has barely any changes to it, so putting it up is pointless.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

model = models.Sequential()



model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) #input shape is image_height, image_width, image_channels
########################^^ output depth
#64/32,(3,3) is the output tensor of each conv2D layer.
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #wfeed this last output tensor into a dense layer

#CLASSIFIER/DENSE LAYER
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#TRAINING THE MODEL

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
                                                         # ^^^^^ comes in from Keras
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size=64)

predictions = model.predict(test)

submission = pd.DataFrame({'ImageId':test['ImageId'],'Label':predictions})
submission.to_csv('submissionMNIST.csv', index=False)
