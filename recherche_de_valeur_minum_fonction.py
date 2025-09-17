from scipy.optimize import minimize

x0 = 1
ab = [24,9]


def volume(x, ab):
    return -(4*x**3-2*(ab[0]+ab[1])*x**2+ab[0]*ab[1]*x)


res = minimize(volume, x0, args=ab, method='nelder-mead', options={'xatol': 1e-8,'disp': True})
print("Optimun value x:", res.x[0])
print("Optimun volume :", -res.fun)
 