from scipy.special import expit

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Régression quadratique
coefficients = np.polyfit(x, y, 2)
quadratic = np.poly1d(coefficients)

# Prédictions
y_pred = quadratic(x)

# Visualisation
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()