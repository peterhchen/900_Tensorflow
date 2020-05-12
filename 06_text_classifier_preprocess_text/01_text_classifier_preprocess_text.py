# Import Python tensorflow package
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)

# Try the encoder
encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'
# encode a string
encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))
# Decode a string
original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))
# Assert the eocode and decode string
# if string1 == string2, nothin ghappend
# if string1 != string2, AssertionError message.
assert original_string == sample_string

# print the encoder word by decode the encode word-by-word
for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

# Explore the Data
# Encode word from the stream of text data
i = 1
for train_example, train_label in train_data.take(3):
  print ('\ni:', i)
  print('train_example[:10].numpy():', train_example[:10].numpy())
  print('train_label.numpy():', train_label.numpy())
  print ('encoder.decode(train_example[:10]):', encoder.decode(train_example[:10]))
  i = i + 1

# Prepare the data for training
BUFFER_SIZE = 1000

# train_batches = (
#     train_data
#     .shuffle(BUFFER_SIZE)
#     .padded_batch(32, padded_shapes=([None],[])))

# test_batches = (
#     test_data
#     .padded_batch(32, padded_shapes=([None],[])))

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32))

test_batches = (
    test_data
    .padded_batch(32))

# Batch shape and label shape
# Each batch has has theiw shape because the user enter text with dynamic length.
print ('\nEach batch has their own shape (batch_size, sequence_length)')
for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

# Build the model
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1)])

print ('\nmodel.summary():')
model.summary()

# Compile Model
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Tran the Model
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

# Evaluate the model
loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

# Plot the result
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# PLot another graph
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()