
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def sarimax_benchmark(train, steps):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    res = model.fit(disp=False)
    return res.forecast(steps)
