import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def f(tau, alpha, C):
    return tau**(2*(alpha-1))*np.exp(2*C*tau**alpha/alpha)

def f_int(t, alpha, C):
    return quad(lambda x: f(x, alpha, C), 1, t)[0]

def a_int(t, alpha, C):
    return (1./alpha)*np.exp(2*C*t**alpha/alpha)*t**(alpha-1)

alpha = 0.2
C = 0.1
ts = np.linspace(1, 100000, 1000)
fs = np.zeros(ts.shape[0])
afs = np.zeros(ts.shape[0])

for i in range(ts.shape[0]):
    fs[i] = f_int(ts[i], alpha, C)
    afs[i] = a_int(ts[i], alpha, C)

plt.plot(ts, fs)
plt.plot(ts, afs)
plt.yscale('log')
plt.show()
