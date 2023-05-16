# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/assignment/C1W4_Assignment.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.preprocessing.image as img

base_dir = "../data/happy_or_sad/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")

print("Sample happy image:")
plt.imshow(img.load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
plt.show()

print("\nSample sad image:")
plt.imshow(img.load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
plt.show()

sample_image = img.load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")
sample_array = img.img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

gen = img.ImageDataGenerator(rescale=1 / 255)
gen = gen.flow_from_directory(
    "../data/happy_or_sad/",
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.75:
            print("\nReached 75% accuracy so cancelling training!")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(gen, steps_per_epoch=2, epochs=20, callbacks=[myCallback()])

# save model in H5 format, to submit for exam
model.save('../saved_models/c1-w4-binary-classifier-with-happy-sad-images.h5')
