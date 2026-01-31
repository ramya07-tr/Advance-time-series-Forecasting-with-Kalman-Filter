
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.F = np.array([[1, 1],[0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[4]])
        self.x = np.zeros((2,1))
        self.P = np.eye(2)

    def fit(self, y):
        self.filtered = []
        for obs in y:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

            # Update
            y_hat = self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K * (obs - y_hat)
            self.P = (np.eye(2) - K @ self.H) @ self.P
            self.filtered.append(self.x[0,0])

    def forecast(self, steps):
        forecasts = []
        x = self.x.copy()
        for _ in range(steps):
            x = self.F @ x
            forecasts.append(x[0,0])
        return np.array(forecasts)
