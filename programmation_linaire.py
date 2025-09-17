import numpy as np
from scipy.optimize import linprog

f=[-40,-60]
A_ub=[[2,1],[1,1],[1,3]]
b_ub=[70,40,90]
bnds=[(0,100),(0,100)]
Res=linprog(f,A_ub,b_ub,bounds=bnds)
print(Res)