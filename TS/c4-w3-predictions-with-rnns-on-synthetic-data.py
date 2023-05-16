# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/assignment/C4W3_Assignment.ipynb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# generate synthetic data
def trend(time, slope=0.0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def generate_time_series():
    time = np.arange(4 * 365 + 1, dtype="float32")

    y_intercept = 10
    series = trend(time, slope=0.005) + y_intercept
    series += seasonality(time, period=365, amplitude=50)
    series += noise(time, noise_level=3, seed=51)

    return time, series


# global vars
@dataclass
class G:
    TIME, SERIES = generate_time_series()
    SPLIT_TIME = 1100
    WINDOW_SIZE = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000


# plot the generated series
def plot_series(time, series, fmt="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], fmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


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


# make windowed dataset
def windowed_dataset(series, window_size=G.WINDOW_SIZE,
                     batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    dataset_ = tf.data.Dataset.from_tensor_slices(series)
    dataset_ = dataset_.window(window_size + 1, shift=1, drop_remainder=True)
    dataset_ = dataset_.flat_map(lambda window: window.batch(window_size + 1))
    dataset_ = dataset_.shuffle(shuffle_buffer)
    dataset_ = dataset_.map(lambda window: (window[:-1], window[-1]))
    dataset_ = dataset_.batch(batch_size).prefetch(1)
    return dataset_


dataset = windowed_dataset(series_train)


# build uncompiled model  to adjust the learning rate first!
def create_uncompiled_model():
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            input_shape=[G.WINDOW_SIZE]),
        tf.keras.layers.SimpleRNN(40, return_sequences=True),
        tf.keras.layers.SimpleRNN(40),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])
    return model_


uncompiled_model = create_uncompiled_model()


try:
    uncompiled_model.predict(dataset)
except:
    print("Incompatible architecture!")
else:
    print("Compatible architecture.")


# adjust the learning rate
def adjust_learning_rate():
    model_ = create_uncompiled_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10 ** (epoch / 20))
    model_.compile(loss=tf.keras.losses.Huber(),
                   optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                   metrics=["mae"])
    history_ = model_.fit(dataset, epochs=100, callbacks=[lr_schedule])
    return history_


lr_history = adjust_learning_rate()     # select lower LR(1e-5)
plt.semilogx(lr_history.history["lr"], lr_history.history["loss"])
plt.axis([1e-6, 1, 0, 30])


# build/train model (w/ this optimal learning rate)
def create_model():
    tf.random.set_seed(51)
    model_ = create_uncompiled_model()
    model_.compile(loss=tf.keras.losses.Huber(),
                   optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                   metrics=["mae"])
    return model_


model = create_model()
history = model.fit(dataset, epochs=50)


# model forecast
def model_forecast(model_, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model_.predict(ds)
    return forecast


rnn_forecast = model_forecast(model, G.SERIES, G.WINDOW_SIZE).squeeze()
rnn_forecast = rnn_forecast[G.SPLIT_TIME - G.WINDOW_SIZE:-1]


plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)
plt.show()


# evaluate metrics: mse: 34.26, mae: 3.70 for forecast (<= 4.5 for valid!)
def compute_metrics(true_series, forecast):
    mse_ = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae_ = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse_, mae_


# mse: 55.28, mae: 4.15 <-- must have MAE <= 4.5!
mse, mae = compute_metrics(series_valid, rnn_forecast)
print(f'mse: {mse:.2f}, mae: {mae:.2f}')


# save model
model.save('../saved_models/c4-w3-predictions-with-rnns-on-synthetic-data.h5')
