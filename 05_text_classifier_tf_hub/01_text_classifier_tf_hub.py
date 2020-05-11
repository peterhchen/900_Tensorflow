# Import Python tensorflow package
import numpy as np

import tensorflow as tf

#pip install -q tensorflow-hub
#pip install -q tfds-nightly
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" \
  if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Download IMDB dataset
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Explore the Data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print ('train_examples_batch:')
print (train_examples_batch)
print ('train_labels_batch:')
print (train_labels_batch)

# Build the Model
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
print('hub_layer(train_examples_batch[:3]:')
print(hub_layer(train_examples_batch[:3]))

# Build the full model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

print ('model.summary():')
print (model.summary())

# Compile the model to use an optimizer and loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Evaluate the Model
results = model.evaluate(test_data.batch(512), verbose=2)
print('results:')
print(results)
print('model.metrics_names:')
print(model.metrics_names)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))