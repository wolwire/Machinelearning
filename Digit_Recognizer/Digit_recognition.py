import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tensorflow import keras
from numpy import argmax
from keras.utils import to_categorical
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
mnist=tf.keras.datasets.mnist
k=mnist.load_data()
(trainx,trainy),(xtest,ytest)=k
trainx.shape
#train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
#train_label=train.pop('label')
#train=np.array(train)
test=np.array(test)
trainy=to_categorical(trainy)
def gen_array(arr):
    img=[]
    for i in arr:
        two_d = (np.reshape(i, (28,28,1)))
        img.append(two_d)
    return np.array(img)
train2=gen_array(trainx)
train2.shape
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=2,
    width_shift_range=0.1,
    height_shift_range=0.1,)
datagen.fit(train2)
i=0
train3=train2
train_label3=trainy
for batch in datagen.flow(train2,trainy,batch_size=10000): 
    train3=np.append(train3,batch[0],axis=0)
    train_label3=np.append(train_label3,batch[1],axis=0)
    i += 1
    if i > 5:
        break
train3.shape
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)))
    plt.imshow(two_d, interpolation='nearest')
    return plt
gen_image(train3[70000,:])
argmax(train_label3[70000]) 
train3=train3/250
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))


model.compile(loss='binary_crossentropy',
            optimizer='RMSprop',
            metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
EPOCHS=15
history = model.fit(
  train3, train_label3,batch_size=256,
  epochs=EPOCHS, validation_split=.2, verbose=2,shuffle=True,callbacks=[learning_rate_reduction])
h=model.predict(gen_array(test)/250)
gen_image(test[4500,:])
argmax(h[4500,:])
t=pd.DataFrame({'ImageId':range(1,len(h)+1),'Label':argmax(h,axis=1)})
t.to_csv('submission.csv',index=False,header=True,mode='a')
