from scipy.special import expit

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 3.5, 4, 4.2])

# Régression logarithmique
coefficients = np.polyfit(np.log(x), y, 1)
a = coefficients[1]
b = coefficients[0]

# Prédictions
y_pred = a + b * np.log(x)

# Visualisation
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()