# setup TensorFlow
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# load public MNIST dataset w/ Keras + convert ints  floating point
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build ML model (by stacking layers)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

#  vector of logits and log-odds scores, per class
predictions = model(x_train[:1]).numpy()

# logits  probabilities, per class
tf.nn.softmax(predictions).numpy()

# define loss function  probabilities closed to random on untrained model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# compile the model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# train model (adjust model params + minimize the loss)
model.fit(x_train, y_train, epochs=5)

# evaluate model (check model perf)
model.evaluate(x_test,  y_test, verbose=2)

# make model return a probability
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
