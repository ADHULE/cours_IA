"This isscriptExample_11_6.py"
#This programimplementsExample11.5usingscikit-learn.”'
# Importthenecessarylibraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Thedatafileisread.APandasdataframeisgenerated:
data =pd.read_csv('diabetes2.csv')
# Thedataframeisreshapedwith:
# ThetargetisinthecolumnOutcome.
x_data =np.array(data['BloodPressure']).reshape(-1, 1)
y_data =np.array(data['Outcome'])
# Themodeliscreatedandthedataisfitted:
model =LogisticRegression()
model.fit(x_data, y_data)
# Theparametersbandthewareprintedout:
print(f'Parameter b:{model.intercept_}')
print(f'Parameter w:{model.coef_}')