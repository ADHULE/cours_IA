from random import *

"This programimplementslinearregression"
# Importthenecessarylibrariesandfunctions
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt

# Thedatapointsaregenerated:
y = np.zeros(100)
x = np.linspace(0, 3, 100)
for i in range(len(x)):
    y[i] = 2 + 4 * x[i] + (-1) ** randint(0, 1) * random()
# Parametersareinitialized:
alpha = 0.001
b = 0
w = 0
e = 1e-6  # Valueofepsilon.
# Valueofmiscomputed:
m = len(y)
# Awhileloopisusedforthecomputationofbandw.
# Gradientdescentalgorithmisdoneinthewhileloop:
while (True):
    # Initializethesumsto0:
    sum0 = 0
    sum1 = 0
    # Scriptcontinuesatnextpage.
    # Scriptcontinuedfrompreviouspage.
    # Aforloopisusedtocomputethesums
    for j in range(len(y)):
        sum0 = sum0 + (b + w * x[j] - y[j])
        sum1 = sum1 + (b + w * x[j] - y[j]) * x[j]
    # Thenewvaluesforbandware:
    bn = b - (2 * alpha / m) * sum0
    wn = w - (2 * alpha / m) * sum1
    # Theconditionstoterminatetheloopare:
    if abs(b - bn) < e and abs(w - wn) < e:
        break
    # Ifconditionsdonothold,updatebandwas:
    else:
        b = bn
        w = wn
# PrintNo.ofiterations,andfinalresults.
# Plotthedataandresults:
print('The totalnumberofiterationsis ', i)
print('Final')
print('b = ', b)
print('w = ', w)
plt.scatter(x, y)
plt.plot(x, b + w * x, color='red')
plt.plot(x, 2 + 4 * x, linestyle='dotted')
plt.show()
