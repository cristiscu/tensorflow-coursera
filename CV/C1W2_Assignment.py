# Sequential Image Classifier on MNIST Dataset
# manually load ../data/mnist.npz
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/assignment/C1W2_Assignment.ipynb

import os
import tensorflow as tf

current_dir = os.getcwd()
data_path = os.path.join(current_dir, "../data/mnist.npz")
(images, labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)
images = images.reshape(60000, 28, 28, 1)
images = images / 255.0


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# callback should fire and interrupt before 9th epoch
history = model.fit(images, labels, epochs=10, callbacks=[MyCallback()])

# save model in H5 format, to submit for exam
model.save('../saved_models/C1W2_Assignment.h5')
