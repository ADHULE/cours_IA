from scipy.special import expit

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données
x = np.linspace(-10, 10, 100)
y = expit(x)

# Visualisation
plt.plot(x, y)
plt.show()