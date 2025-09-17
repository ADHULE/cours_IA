import numpy as np
from cvxopt import matrix, solvers

A=matrix(np.array([[1.0,0.0],[0.0,0.0]]))
p=matrix(np.array([3.0,4.0]))
B=matrix(np.array([[-1.0,0],[0,-1.0],[1.0,-3.0],[2.0,5.0],[3.0,4.0]]))
b=matrix(np.array([0.0,0.0,15.0,100.0,80.0]))

solution=solvers.qp(A,p,B,b)
x=solution['x']
ObjFun=solution['primal objective']
print('The valuesofxandyare:')
print("The valueofxis: ",x[0])
print("The valueofyis: ",x[1])
print("The valueoftheobjectivefunctionis: ",ObjFun)