import tensorflow as tf
mnist = tf.keras.datasets.mnist

# keras cache data in
# "C:/Users/14088/.keras/dataset/mnist.npz"
# They are store in binary format.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print ('x_train[:2, :2]:')
print (x_train[:2, :2])
print ('x_test[:2, :2]:')
print (x_test[:2, :2])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print ('predictions:')
print (predictions)

tf.nn.softmax(predictions).numpy()
print ('tf.nn.softmax(predictions).numpy():')
print (tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print ('loss_fn:')
print (loss_fn)

print('loss_fn(y_train[:1], predictions).numpy():')
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=10)
print ('model.evaluate(x_test,  y_test, verbose=2):')
score = model.evaluate(x_test,  y_test, verbose=2)
print ('score:')
print (score)

# If you want your model to return a probability, 
# you can wrap the trained model, 
# and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
print ('probability_model(x_test[:5]):')
print (probability_model(x_test[:5]))