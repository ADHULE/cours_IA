"This isscriptExample_11_7.py"
"This programimplementslogisticregressionwithmultiple"
"features."
# Thenecessarylibrariesandmethodsareimported:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Scriptcontinuesnextpage.
# Scriptcontinuedfrompreviouspage.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Thedataisreadinandtheheadingisprinted:
data =pd.read_csv('diabetes2.csv')
print(data.head(10))
# Thedataissplit.Thedatahas9columns,
# thelastoneistheOutcome.
#The trainingsliceis85%.
X_train, X_test,y_train,y_test=train_test_split\
      (data.iloc[:, 0:7],data['Outcome'], test_size=0.15)
# Themodelisdefinedandthefittingisdone:
model =LogisticRegression()
model.fit(X_train, y_train)
# Thepredictioniscomputedandprinted:
prediction =model.predict(X_test)
print(prediction)
# Thescoreiscomputed:
a =model.score(X_test,y_test)
print(a)
# Finally,theparametersforthemodelareprinted:
print(f'Parameter b:{model.intercept_}')
print(f'Parameter w:{model.coef_}')