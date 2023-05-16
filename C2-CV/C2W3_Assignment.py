# Transfer Learning w/ Horses/Humans on InceptionV3
# must have the InceptionV3 weights already in ../data/ folder
# must already have ../data/horse-or-human.zip and ../data/validation-horse-or-human.zip
# this is a very large dataset (try on GPUs) --> "Allocation of 179437568 exceeds 10% of free system memory." errors!
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W3/assignment/C2W3_Assignment.ipynb

import os, zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

# unpack Horse/Human archives into training/validation subfolders
test_local_zip = '../data/horse-or-human.zip'
train_dir = '../data/horse-or-human/training'
zip_ref = zipfile.ZipFile(test_local_zip, 'r')
zip_ref.extractall(train_dir)
train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')

val_local_zip = '../data/validation-horse-or-human.zip'
validation_dir = '../data/horse-or-human/validation'
zip_ref = zipfile.ZipFile(val_local_zip, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

# create image generators for training/validation
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  train_datagen = ImageDataGenerator(rescale=1./255.,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)
  train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode='binary',
                                                      target_size=(150, 150))

  validation_datagen = ImageDataGenerator(rescale=1.0/255.)
  validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(150, 150))

  return train_generator, validation_generator

train_generator, validation_generator = \
  train_val_generators(train_dir, validation_dir)

# import the Inception model
def create_pre_trained_model(local_weights_file):
  pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                  include_top=False,
                                  weights=None)

  pre_trained_model.load_weights(local_weights_file)
  for layer in pre_trained_model.layers:
    layer.trainable = False
  return pre_trained_model


local_weights_file = '../data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = create_pre_trained_model(local_weights_file)
pre_trained_model.summary()


# stop training when accuracy reached
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy') > 0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


# create and train new model using transfer learning
def create_final_model(pre_trained_model, last_output):
  x = tf.keras.layers.Flatten()(last_output)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(
      inputs=pre_trained_model.input,
      outputs=x)

  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

last_output = pre_trained_model.get_layer('mixed7').output
model = create_final_model(pre_trained_model, last_output)

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    verbose=2,
                    callbacks=[myCallback()])

# should break loop and reach accuracy 99.9%+ after 11 epochs!
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()


# save model in H5 format, to submit for exam
model.save('../saved_models/C2W3_Assignment.h5')
