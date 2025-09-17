#Régression Polynomiale

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Régression polynomiale
coefficients = np.polyfit(x, y, 2)
polynomial = np.poly1d(coefficients)

# Prédictions
y_pred = polynomial(x)

# Visualisation
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()