"This isscriptExample_11_8.py"
"This programimplementslogisticregressionwithmultiple"
"features."
# Importthenecessarylibraries:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Thedata is read
iris =sns.load_dataset('iris')
# Thetargetvaluesareconvertedtonumericvalues:
#iris['species'] =iris['species'].replace('setosa' == 1, \
 #    'versicolor' ==2,'virginica'==3)
# Scriptcontinuesnextpage.
# Scriptcontinuedfrompreviouspage.
# Thedataissplitfortrainingwith85%:
X_train, X_test,y_train,y_test=train_test_split\
       (iris[['sepal_length', 'sepal_width', 'petal_length', \
       'petal_width']], iris['species'],train_size =0.85)
# Themodelisdefinedandthefittingisimplementedwith:
model =linear_model.LogisticRegression()
model.fit(X_train, y_train)
# ApredictionisrunforX_test:
# Totestthemodel,thefunctionpredictisusedas:
prediction =model.predict(X_test)
print(prediction)
# Themodelcanbemeasuredwiththescoremethodas:
a =model.score(X_test,y_test)
print(a)
# Theconfusionmatrixiscomputedandprinted:
cm =confusion_matrix(y_test,prediction)
print(cm)
# Theconfusionmatrixisplottedwith:
plt.figure(figsize =(5,4))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted value')
plt.ylabel('Actual Value')
# Theplotofpetalwidthvs.petallenghtisproduced:
sns.scatterplot(x ='petal_length', y ='petal_width', \
      data =iris,hue= 'species')
# Theplotsaredisplayed:
plt.show()