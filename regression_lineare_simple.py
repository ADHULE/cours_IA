import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 9, 11])

# Régression linéaire
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Prédictions
y_pred = slope * x + intercept


# Visualisation
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()


