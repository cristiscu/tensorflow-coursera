# Binary Classifier on Dogs vs Cats
# must already have images in ../data/cats_and_dogs/ folder, training/validation and cats/dogs subfolders
# this is a very large dataset (try on GPUs) --> "Allocation of 179437568 exceeds 10% of free system memory." errors!
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/assignment/C2W2_Assignment.ipynb

import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

# create directories
root_dir = '../data/cats_and_dogs/'

train_dir = os.path.join(root_dir, 'training/')
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
train_cat_dir = os.path.join(root_dir, 'training/cats/')
os.makedirs(train_cat_dir)
train_cat_names = os.listdir(train_cat_dir)
train_dog_dir = os.path.join(root_dir, 'training/dogs/')
os.makedirs(train_dog_dir)
train_dog_names = os.listdir(train_dog_dir)

val_dir = os.path.join(root_dir, 'validation/')
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
val_cat_dir = os.path.join(root_dir, 'validation/cats/')
os.makedirs(val_cat_dir)
val_cat_names = os.listdir(val_cat_dir)
val_dog_dir = os.path.join(root_dir, 'validation/dogs/')
os.makedirs(val_dog_dir)
val_dog_names = os.listdir(val_dog_dir)

for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))


# split data and copy image files into the training/validation directories
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE_DIR):
        file_ = SOURCE_DIR + filename
        if os.path.getsize(file_) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE_DIR + filename
        destination = TRAINING_DIR + filename
        copyfile(this_file, destination)
    for filename in testing_set:
        this_file = SOURCE_DIR + filename
        destination = VALIDATION_DIR + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "../data/cats_and_dogs/Cat/"
DOG_SOURCE_DIR = "../data/cats_and_dogs/Dog/"
TRAINING_DIR = "../data/cats_and_dogs/training/"
VALIDATION_DIR = "../data/cats_and_dogs/validation/"
TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")
TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

if len(os.listdir(TRAINING_CATS_DIR)) > 0:
    for file in os.scandir(TRAINING_CATS_DIR):
        os.remove(file.path)
if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
    for file in os.scandir(TRAINING_DOGS_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
    for file in os.scandir(VALIDATION_CATS_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
    for file in os.scandir(VALIDATION_DOGS_DIR):
        os.remove(file.path)

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

print(f"\n\nOrig cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images")
print(f"Orig dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")
print(f"{len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"{len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"{len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"{len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")


# create image generators, w/ data augmentation for training
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DIR,
        batch_size=128,
        class_mode='binary',
        target_size=(150, 150))

    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        batch_size=32,
        class_mode='binary',
        target_size=(150, 150))

    return train_generator, validation_generator


train_generator, validation_generator =\
    train_val_generators(TRAINING_DIR, VALIDATION_DIR)


# create and train a model for binary classification w/ CNN
def create_model():
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model_


model = create_model()
history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)

# visualize and estimate model performance
# must have: training/validation accuracy >= 80%
# and testing accuracy > training accuracy or max 5% diff
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()

# save model in H5 format, to submit for exam
model.save('../saved_models/C2W2_Assignment.h5')
