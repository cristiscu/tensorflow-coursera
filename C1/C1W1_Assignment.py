# Predict House Prices
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W1/assignment/C1W1_Assignment.ipynb

import numpy as np
import tensorflow as tf

# create model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# train model
xs = np.array([1, 2, 3, 4, 5, 6], dtype=int)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
model.fit(xs, ys, epochs=500)

# predict next house price --> expected close to 4 ([4.050522])
print(model.predict([7.0])[0])

# save model in H5 format, to submit for exam
model.save('../saved_models/C1W1_Assignment.h5')
