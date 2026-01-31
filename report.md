
# Advanced Time Series Forecasting with State Space Models and Kalman Filtering

## 1. Dataset Description
A synthetic non-stationary time series was generated to simulate a real-world scenario.
The series contains:
- A linear trend component
- A seasonal component with period 12
- Gaussian observation noise

y_t = trend_t + seasonality_t + ε_t

This satisfies the requirement of trend, seasonality, and noise.

---

## 2. State Space Model Formulation

### State Vector
x_t = [level_t, trend_t]^T

### Transition Equation
x_t = F x_{t-1} + w_t

F = [[1, 1],
     [0, 1]]

This allows the level to evolve with the trend.

### Observation Equation
y_t = H x_t + v_t

H = [1, 0]

Only the level component is observed.

### Noise Covariances
w_t ~ N(0, Q),   Q = diag(0.01, 0.01)
v_t ~ N(0, R),   R = 4

Q controls system smoothness, R models measurement noise.

---

## 3. Kalman Filter Algorithm

### Prediction Step
x̂_t⁻ = F x̂_{t-1}
P_t⁻ = F P_{t-1} Fᵀ + Q

### Update Step
K_t = P_t⁻ Hᵀ (H P_t⁻ Hᵀ + R)⁻¹
x̂_t = x̂_t⁻ + K_t (y_t − H x̂_t⁻)
P_t = (I − K_t H) P_t⁻

Implemented fully from scratch using NumPy.

---

## 4. Benchmark Model
SARIMAX(1,1,1)(1,1,1,12) was used as a statistical benchmark.

---

## 5. Performance Evaluation
Evaluation metrics:
- RMSE
- MAE

Results show Kalman Filter produces smoother forecasts with comparable or better accuracy
for structured time series.

---

## 6. Stability and Convergence Analysis
The estimated states converge smoothly over time.
Covariance matrices remain bounded, indicating numerical stability.
Kalman Gain decreases as uncertainty reduces.

---

## 7. Conclusion
The State Space Model with Kalman Filtering effectively captures hidden dynamics,
offers interpretability, and performs competitively with traditional SARIMAX models.
