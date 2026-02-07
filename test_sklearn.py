import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Simple dummy data
X = np.random.rand(100, 5)
y = np.random.rand(100)

try:
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    print("Sklearn HistGradientBoostingRegressor works!")
except Exception as e:
    print(f"Sklearn failed: {e}")
