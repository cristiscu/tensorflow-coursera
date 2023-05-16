# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/assignment/C4W4_Assignment.ipynb
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass


def parse_data_from_file(filename):
    times = []
    temperatures = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        i = 0
        for row in reader:
            times.append(i)
            i = i + 1
            temperatures.append(float(row[1]))
    return times, temperatures


# global vars
@dataclass
class G:
    TEMPERATURES_CSV = '../data/daily-min-temperatures.csv'
    times, temperatures = parse_data_from_file(TEMPERATURES_CSV)
    TIME = np.array(times)
    SERIES = np.array(temperatures)
    SPLIT_TIME = 2500
    WINDOW_SIZE = 64
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000


# plot loaded series
def plot_series(time, series, fmt="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], fmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


plt.figure(figsize=(10, 6))
plot_series(G.TIME, G.SERIES)
plt.show()


# split training/validation data
def train_val_split(time, series, time_step=G.SPLIT_TIME):
    time_train_ = time[:time_step]
    series_train_ = series[:time_step]
    time_valid_ = time[time_step:]
    series_valid_ = series[time_step:]
    return time_train_, series_train_, time_valid_, series_valid_


time_train, series_train, time_valid, series_valid =\
    train_val_split(G.TIME, G.SERIES)


# get windowed data
def windowed_dataset(series, window_size=G.WINDOW_SIZE,
                     batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


train_set = windowed_dataset(series_train, window_size=G.WINDOW_SIZE,
                             batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)


# create uncompiled model  to adjust learning_rate
def create_uncompiled_model():
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                               padding="causal", activation="relu",
                               input_shape=[G.WINDOW_SIZE, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model_


uncompiled_model = create_uncompiled_model()

try:
    uncompiled_model.predict(train_set)
except:
    print("Incompatible architecture.")
else:
    print("Compatible architecture.")


# adjust the learning rate
def adjust_learning_rate(dataset):
    model_ = create_uncompiled_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10 ** (epoch / 20))
    model_.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        metrics=["mae"])

    # this can take a while without GPU!
    history_ = model_.fit(dataset, epochs=100, callbacks=[lr_schedule])
    return history_


lr_history = adjust_learning_rate(train_set)

# select optimal learning rate from graph below (lowest point)  1e-3
plt.semilogx(lr_history.history["lr"], lr_history.history["loss"])
plt.axis([1e-4, 10, 0, 10])


# build/train model w/ determined learning_rate
def create_model():
    model_ = create_uncompiled_model()
    model_.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
        metrics=["mae"])
    return model_


model = create_model()
history = model.fit(train_set, epochs=50)


# faster model forecasts
def model_forecast(model_, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model_.predict(ds)
    return forecast


rnn_forecast = model_forecast(model, G.SERIES, G.WINDOW_SIZE).squeeze()
rnn_forecast = rnn_forecast[G.SPLIT_TIME - G.WINDOW_SIZE:-1]

# plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)


# evaluate the forecast
def compute_metrics(true_series, forecast):
    # error: InvalidArgumentError - Incompatible shapes: [1150,64] vs. [1150] [Op:SquaredDifference]
    mse_ = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae_ = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse_, mae_


# mse: 55.28, mae: 4.15 <-- must have MSE<=6 and MAE<=2!
mse, mae = compute_metrics(series_valid, rnn_forecast)
print(f'mse: {mse:.2f}, mae: {mae:.2f}')

# save model in H5 format
model.save('../saved_models/c4-w4-predictions-on-daily-min-temps-in-melbourne.h5')
