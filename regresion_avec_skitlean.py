# Example_11_3.pyLinearregressionwithscikit-learn.
# Therequiredlibrariesareimported:
from sklearn.linear_model import LinearRegression
from random import randint, random
import matplotlib.pyplot as plt
import numpy as np

# Thedatapointsaregenerated:
y = np.zeros(100)
x = np.linspace(0, 3, 100)
for i in range(len(x)):
    y[i] = 2 + 4 * x[i] + (-1) ** randint(0, 1) * random()
# Scriptcontinuesatnextpage.
# Scriptcontinuedfrompreviouspage.
# Thearraysxandyhavetobereshapedascolumnarrays.
# Thisisrequiredbysklearn:
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
# Themodelis
lin_reg = LinearRegression()
# Thelinearregressionisimplementedwiththefitmethodas:
lin_reg.fit(x, y)
# They-axisinterceptandtheslopeare:
b = lin_reg.intercept_
w = lin_reg.coef_
# Linearregressionusingpartofdatafortraining
# andpartfortesting.
# Thenecessarylibrariesareimported:
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Thedataissplitted,80%fortrainigand20%fortesting:
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=23)
# Themodeliscreated:
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
# Coefficientsbandwandm
bs = lin_reg.intercept_
ws = lin_reg.coef_
# Scriptcontinuesatnextpage.
# Scriptcontinuedfrompreviouspage.
# Thesquarederroriscomputed.Thepredictedvaluesare:
yps = ws * x_train + bs
# Squarederror
sqs = r2_score(y_train, yps)
# Displayingtheresults
# Thecoefficientsb,w,andthesquarederror
# whenthedataaresplittedareprinted:
print('The y-axisinterceptandtheslopeare:\
      \n', b[0], w[0][0])
print('The value of r∧2 is: ', sqs)
# Thesquarederrorwithouttrainingsplittingis:
yp = w * x + b
sqs = r2_score(y, yp)
# Thesquarederrorisprinted:
print('The valuewithoutsplittingofr∧2 is: ', sqs)
# Thecoefficientsbs,ws,andthesquarederrorareprinted.
# Thecoefficientsb0,w1,andthesquarederrorareprinted:
print('The y-axisinterceptandtheslopewhendata\
     is splittedare:\n', bs[0], ws[0][0])
print('The valueofr∧2 is: ', sqs)
# Thedatapointsandthestraightlinefrom
# thelinearregressionareplotted:
plt.scatter(x, y)
plt.plot(x, b + w * x, color='black')
plt.plot(x, bs + ws * x, color='red')
plt.show()
