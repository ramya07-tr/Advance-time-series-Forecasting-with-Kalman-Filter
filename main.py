
import numpy as np
import pandas as pd
from kalman import KalmanFilter
from benchmark import sarimax_benchmark
from metrics import rmse, mae

# Generate synthetic time series
np.random.seed(42)
n = 200
t = np.arange(n)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, n)
y = trend + seasonal + noise

# Train-test split
train_size = int(0.8 * n)
y_train, y_test = y[:train_size], y[train_size:]

# Kalman Filter
kf = KalmanFilter()
kf.fit(y_train)
kf_forecast = kf.forecast(len(y_test))

# Benchmark
sarimax_forecast = sarimax_benchmark(y_train, len(y_test))

print("Kalman RMSE:", rmse(y_test, kf_forecast))
print("Kalman MAE :", mae(y_test, kf_forecast))
print("SARIMAX RMSE:", rmse(y_test, sarimax_forecast))
print("SARIMAX MAE :", mae(y_test, sarimax_forecast))
