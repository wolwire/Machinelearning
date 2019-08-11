from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from zipfile import ZipFile
print(tf.__version__)
from os.path import join
path_to_file='../input/Admission_Predict.csv'
data = pd.read_csv (path_to_file)
print(type(data))
data.pop('Serial No.')
train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)
plt.plot(data['GRE Score'][0:399],data['Chance of Admit '][0:399],'ro')
print(train_dataset)
train_labels=train_dataset.pop('Chance of Admit ')
test_labels=test_dataset.pop('Chance of Admit ')
#(train_dataset-train_dataset.mean())/train_dataset.std()
normdata=(train_dataset-train_dataset.mean())/train_dataset.std()
normdata;
t_normdata=(test_dataset-train_dataset.mean())/train_dataset.std()
model = keras.Sequential()    
model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1))
optimizer = keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error'])
model.summary()
example_batch = t_normdata[:10]
example_result=model.predict(example_batch);
EPOCHS=200
history = model.fit(
  normdata, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,.2])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,.2])
  plt.legend()
  plt.show()


plot_history(history)
test_predictions = model.predict(t_normdata).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
tp={'GRE Score':[330],'TOEFL Score':[110],'University Rating':[2],'SOP':[3.5],'LOR ':[3.5],'CGPA':[6.78],'Research':[1]}
tp2=pd.DataFrame(tp)
tp2=(tp2-train_dataset.mean())/train_dataset.std()
test_predictions = model.predict(tp2).flatten()
test_predictions
