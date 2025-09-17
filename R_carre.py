from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Modèles
models = {
    "Linéaire": np.poly1d(np.polyfit(x, y, 1)),
    "Quadratique": np.poly1d(np.polyfit(x, y, 2)),
    "Cubique": np.poly1d(np.polyfit(x, y, 3))
}

# Comparaison
for name, model in models.items():
    y_pred = model(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")