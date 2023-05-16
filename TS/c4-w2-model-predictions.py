# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/assignment/C4W2_Assignment.ipynb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass


# generate time series (synthetic data)
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


# global variables + generate time series
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


# split into training/validation sets
def train_val_split(time, series, time_step=G.SPLIT_TIME):
    time_t = time[:time_step]
    series_t = series[:time_step]
    time_v = time[time_step:]
    series_v = series[time_step:]
    return time_t, series_t, time_v, series_v


time_train, series_train, time_valid, series_valid = \
    train_val_split(G.TIME, G.SERIES)


# generate windowed datasets
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
test_dataset = windowed_dataset(series_train,
                                window_size=1, batch_size=5, shuffle_buffer=1)


# build/train model
def create_model(window_size=G.WINDOW_SIZE):
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model_.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    return model_


model = create_model()
model.fit(dataset, epochs=100)


# evaluate the forecast
def generate_forecast(series=G.SERIES,
                      split_time=G.SPLIT_TIME, window_size=G.WINDOW_SIZE):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(
            series[time:time + window_size][np.newaxis]))
    forecast = forecast[split_time - window_size:]
    results = np.array(forecast)[:, 0, 0]
    return results


dnn_forecast = generate_forecast()


# plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, dnn_forecast)
plt.show()


# calculate MSE/MAE - MSE should be here <= 30
def compute_metrics(true_series, forecast):
    mse_ = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae_ = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse_, mae_


# mse: 31.44, mae: 3.42 <-- must have MSE <= 30!
mse, mae = compute_metrics(series_valid, dnn_forecast)
print(f'mse: {mse:.2f}, mae: {mae:.2f}')


# save model in H5 format, to submit for exam
model.save('../saved_models/c4-w2-model-predictions.h5')
