# Import Python tensorflow package
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Print Tensorflow veriosn 
print(tf.__version__)   # 2.2.0

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Define function
def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Get data_path from Auto MPG (Mile per Gallon) Dataset
dataset_path = keras.utils.get_file("auto-mpg.data", 
"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print ('\ndataset_path:')
print (dataset_path)

# import dataset using panda
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print ('\ndataset.tail():')
print (dataset.tail())

# Clean the data: There are some N/A rows [isna()].
print ('\ndataset.isna().sum():')
print (dataset.isna().sum())

# Drop those rows: reset the new dataset.
dataset = dataset.dropna()
# The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
# print ('\ndataset:')
# print (dataset)
print ('\ndataset.tail():')
print (dataset.tail())

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print ('\ntrain_dataset[:10]:')
print (train_dataset[:10])
print ('\ntest_dataset[:10]:')
print (test_dataset[:10])

# Inspect the data
print ('\nsns.pairplot(train_dataset):')
print (sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], 
diag_kind="kde"))

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print('\ntrain_stats:')
print(train_stats)

# Split features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
print ('\ntrain_labels[:10]:')
print (train_labels[:10])
print ('\ntest_labels[:10]:')
print (test_labels[:10])
# Normalize the data
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print ('\ntrain_dataset[:10]:')
print (train_dataset[:10])
print ('\nnormed_train_data[:10]:')
print (normed_train_data[:10])
print ('\ntest_dataset[:10]:')
print (test_dataset[:10])
print ('\nnormed_test_dataset[:10]:')
print (normed_test_data[:10])

# Build the Model
model = build_model()

# Inspect the Model Summary
print ('\nmodel.summary: ')
model.summary()

# Try out the model. 
# Take a batch of 10 examples from the training data and call model.predict on it.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print ('\nexample_batch:')
print (example_batch)
print ('\nexample_result:')
print (example_result)

# Train the model for 1000 epochs, and record the training 
# and validation accuracy in the history object.
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

# Visual model training
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\nhist[:10]:')
print(hist[:10])
print("\nhist['epoch'][:10]:")
print(hist['epoch'][:10])

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mae")

plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.show()

# Build model
model = build_model()
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Make Prediction
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# Plot histogram
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
