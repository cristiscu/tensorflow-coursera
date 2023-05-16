# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W1/assignment/C4W1_Assignment.ipynb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


# generate data  w/ trend + seasonality + noise
TIME = np.arange(4 * 365 + 1, dtype="float32")

y_intercept = 10
SERIES = trend(TIME, slope=0.01) + y_intercept
SERIES += seasonality(TIME, period=365, amplitude=40)
SERIES += noise(TIME, noise_level=2, seed=42)


# plot the data
def plot_series(time, series, fmt="_", title="", label=None, start=0, end=None):
    plt.plot(time[start:end], series[start:end], fmt, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)


plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)
plt.show()

# split data  into training + validation
SPLIT_TIME = 1100
time_train = TIME[:SPLIT_TIME]
series_train = SERIES[:SPLIT_TIME]
time_valid = TIME[SPLIT_TIME:]
series_valid = SERIES[SPLIT_TIME:]

plt.figure(figsize=(10, 6))
plot_series(time_train, series_train, title="Training")
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, title="Validation")
plt.show()


# evaluation metrics MSE/MAE (on dummy series for testing)
def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    print(f"mse: {mse}, mae: {mae} …\n")
    return mse, mae


zeros, ones = np.zeros(5), np.ones(5)
compute_metrics(zeros, ones)                # 1.0, 1.0
compute_metrics(ones, ones)                 # 0.0, 0.0

# (1) naive forecasting  check MSE/MAE, lags 1 step behind (zoom at the end)
naive_forecast = SERIES[SPLIT_TIME - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, label="validation set")
plot_series(time_valid, naive_forecast, label="naive forecast")
plt.show()


# (2) moving average (w/ window_size)  MSE/MAE worse than naive forecast
def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


moving_avg = moving_average_forecast(SERIES, window_size=30)
moving_avg = moving_avg[1100 - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, moving_avg)

# (3) differencing = remove trend/seasonality from moving average  better
diff_series = (SERIES[365:] - SERIES[:-365])
diff_time = TIME[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)
diff_moving_avg = diff_moving_avg[SPLIT_TIME - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[1100 - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

past_series = SERIES[SPLIT_TIME - 365:-365]
diff_moving_avg_plus_past = past_series + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

# Perform the correct split of SERIES
smooth_past_series = moving_average_forecast(SERIES[SPLIT_TIME - 370:-360], 10)
diff_moving_avg_plus_smooth_past = smooth_past_series + diff_moving_avg


plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

# MSE=11.834952, MAE=2.1856763
print(tf.keras.metrics.mean_squared_error(series_valid, diff_moving_avg_plus_smooth_past).numpy())
print(tf.keras.metrics.mean_absolute_error(series_valid, diff_moving_avg_plus_smooth_past).numpy())
