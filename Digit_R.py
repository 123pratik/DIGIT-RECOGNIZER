#IMPORTING THE LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORTING THE TRAIN DATASET
training_set = pd.read_csv('train.csv')
print(training_set.shape)

#USING COUNTER TO GET THE UNIQUE NUMBERS OF LABEL
from collections import Counter
Unique = Counter(training_set['label'])

#VISUALIZING
sns.countplot(training_set['label'])

#IMPORTING THE TEST DATASET
test_set = pd.read_csv('test.csv')
print(test_set.shape)

training_set.head()

a_train = training_set.iloc[:, 1:].values
b_train = training_set.iloc[:, 0].values
a_test = test_set.values

plt.figure(figsize = (10, 8))
a, b = 9, 3
for i in range(27):
    plt.subplot(b, a, i+1)
    plt.imshow(a_train[i].reshape((28, 28)))
plt.show()

#NORMALISING
a_train = a_train/255.0
a_test = a_test/255.0
b_train

#RESHAPING TO THE KERAS EXPECTED FORMAT
A_train = a_train.reshape(a_train.shape[0], 28, 28, 1)
A_test = a_test.reshape(a_test.shape[0], 28, 28, 1)

#IMPORTING THE KERAS LIBRARIES
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
batch_size = 32
epochs = 100
num_classes = 10
input_shape = (28, 28, 1)

#SPILTTING THE DATA AND USING TO CATEGORICAL WHICH WILL CONVERT INTEGER TO BINARY MATRIX
from sklearn.model_selection import train_test_split
b_train = keras.utils.to_categorical(b_train, num_classes)
A_train, A_val, B_train, B_val = train_test_split(A_train, b_train, test_size = 0.2)

#INITIALIZING THE NEURAL NETWORK
Digit = Sequential()

#ADDING CONVOLUTION, MAX-POOL LAYERS AND USING DROPOUT TO MINIMIZE THE OVERFITTING
Digit.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

Digit.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

Digit.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

#FLATTENING
Digit.add(Flatten())

#ADDING THE LAYERS
Digit.add(Dense(units = 128, activation = 'relu'))

#USING BATCHNORMALIZATION TO MAINTAIN MEAN ACTIVATION CLOSE TO 0 AND STANDARD DEVIATION CLOSE TO 1
Digit.add(BatchNormalization())
Digit.add(Dropout(rate = 0.1))

#ADDING THE OUTPUT LAYER
Digit.add(Dense(num_classes, activation = 'sigmoid'))

#COMPILING THE MODEL
Digit.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#USING REDUCE LEARNING RATE ON PLATEAU WHICH WILL DROP THE METRIC WHICH HAS STOPPED IMPROVING
learning_rate_reduction = ReduceLROnPlateau(monitor ='val_loss', 
                                            patience = 10, 
                                            verbose= 0, 
                                            factor = 0.1, 
                                            min_lr = 0.0001)

#IMAGE AUGMENTATION
IDG = ImageDataGenerator(rotation_range=15, 
                         zoom_range = 0.1,
                         width_shift_range=0.1,  
                         height_shift_range=0.1)
Digit.summary()

#FITTING THE MODEL
IDG.fit(A_train)
h = Digit.fit_generator(IDG.flow(A_train, B_train, batch_size = batch_size),
                                 epochs = epochs, validation_data = (A_val, B_val),
                                                                     steps_per_epoch = a_train.shape[0] // batch_size,
                                                                     callbacks = [learning_rate_reduction],)
#CALCULATING THE LOSS AND ACCURACY
final_loss, final_acc = Digit.evaluate(A_val, B_val)
print('Final loss : {0:.6f},  final accuracy : {1:.6f}'.format(final_loss, final_acc))

#PREDICTING THE RESULT
result = Digit.predict_classes(A_val)



