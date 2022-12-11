import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import math


# Problem 1
# p(x) = f[a] + f[a,a](x-a) + ... + f[a,...,a](x-a)^p
#                                     (p+1) a's
#        + f[a,...,a,b](x-a)^(p+1) + ... + f[a,...,a,b,...,b] (x-a)^(p+1) (x-b)^p
#                                          (p+1) a's; p b's

def sin(x, p):
    # x: nodes to estimate
    # p: the degree of the Hermite interpolant

    # Step 1: determine which subinterval x belongs to
    x = x % (2 * np.pi)
    h = 1/4 * np.pi
    i = x // h
    a = i * h
    b = (i+1) * h


    # Step 2: call GDD(p, a, b)
    #         -> return coefficient array [f[a], f[a,a], ..., f[a,...,a], 
    #                                      f[a,...,a,b], ..., f[a,...,a,b,...,b]]
    coefficient_array = gdd(p ,a, b)


    # Step 3: compute the result
    res = coefficient_array[0]
    for i in range(1, p+1):
        # i = 0   , 1          , ..., p
        #     f[a], f[a,a](x-a), ..., f[a,...,a](x-a)^p
        res += coefficient_array[i] * (x-a)**i

    # f[a,...,a,b](x-a)^(p+1)
    res += coefficient_array[p+1] * (x-a)**(p+1) * (x-b)

    for i in range(p+2, 2*p+2):
        # i = p+2                         , ..., 2p+1
        #     f[a,...,a,b](x-a)^(p+1)(x-b), ..., f[a,...,a,b,...,b] (x-a)^(p+1) (x-b)^p
        res += coefficient_array[i] * (x-a)**(p+1) * (x-b)**(i-p-1)

    return res


# compute the coefficient array [f[a], f[a,a], ..., f[a,...,a], 
#                                f[a,...,a,b], ..., f[a,...,a,b,...,b]]
def gdd(p, a, b):
    arr = np.array([a] * (p+1) + [b] * (p+1))
    res = np.array([])
    for i in range(len(arr)):
        res = np.append(res, dp(arr[:i+1]))
    return res
    

# compute f[arr]
# e.g. arr = [a,a,b,b], compute f[a,a,b,b]
def dp(arr):
    if len(arr) == 1:
        return np.sin(arr[0])
    
    if arr[0] == arr[-1]:
        p = len(arr) - 1
        return differentiate(arr[0], p) / math.factorial(p) 

    left = dp(arr[:-1])
    right = dp(arr[1:])
    return (right - left) / (arr[-1] - arr[0])


# compute the pth derivative of sin(x) at x = a
def differentiate(a, p):
    if p % 4 == 0:
        return np.sin(a)
    elif p % 4 == 1:
        return np.cos(a)
    elif p % 4 == 2:
        return -np.sin(a)
    elif p % 4 == 3:
        return -np.cos(a)



# Problem 2
plt.subplots(2, 2, figsize=(10, 10))
plt.suptitle("Approximation Error vs. Degree of Interpolant", y=0.95, fontsize="xx-large")
i = 1
x = np.linspace(0, 2*np.pi, 50)
errors = np.array([])
degrees = np.array([2, 4, 6, 8])
for p in degrees:
    y = np.array([])
    for element in x:
        y = np.append(y, sin(element, p))
    error = np.array(abs(np.sin(x)-y))
    errors = np.append(errors, max(error))
    plt.subplot(2, 2, i)
    i += 1
    plt.semilogy(x, error)
    plt.title(f"Degree={str(p)}")
    plt.xlabel("Value x")
    plt.ylabel("Error")
plt.tight_layout()
plt.subplots_adjust(top=0.85) 
plt.show()

plt.semilogy(degrees, errors)
plt.title("Approximation Error vs. Degree of Interpolant")
plt.xlabel("Degrees")
plt.ylabel("Errors")
plt.show()


# Bonus Problem
# L1 norm
N = 200
f_L1 = 4
nodes = np.linspace(0, 2*np.pi, N)
E_N_L1 = 0
for i in range(len(nodes)-1):
    val1 = abs(np.sin(nodes[i]) - sin(nodes[i], 5))
    val2 = abs(np.sin(nodes[i+1]) - sin(nodes[i+1], 5))
    E_N_L1 += val1 + val2
E_N_L1 *= 2*np.pi/N / f_L1
print(E_N_L1)

nodes = np.linspace(0, 2*np.pi, 2*N)
E_2N_L1 = 0
for i in range(len(nodes)-1):
    val1 = abs(np.sin(nodes[i]) - sin(nodes[i], 5))
    val2 = abs(np.sin(nodes[i+1]) - sin(nodes[i+1], 5))
    E_2N_L1 += val1 + val2
E_2N_L1 *= np.pi/N / f_L1
print(E_2N_L1)


# L2 norm
N = 200
f_L2 = np.sqrt(np.pi)
nodes = np.linspace(0, 2*np.pi, N)
E_N_L2 = 0
for i in range(len(nodes)-1):
    val1 = (np.sin(nodes[i]) - sin(nodes[i], 5))**2
    val2 = (np.sin(nodes[i+1]) - sin(nodes[i+1], 5))**2
    E_N_L2 +=  val1 + val2
E_N_L2 = np.sqrt(np.pi/(2*N) * E_N_L2) / f_L2
print(E_N_L2)

nodes = np.linspace(0, 2*np.pi, 2*N)
E_2N_L2 = 0
for i in range(len(nodes)-1):
    val1 = (np.sin(nodes[i]) - sin(nodes[i], 5))**2
    val2 = (np.sin(nodes[i+1]) - sin(nodes[i+1], 5))**2
    E_2N_L2 +=  val1 + val2
E_2N_L2 = np.sqrt(np.pi/(2*N) * E_2N_L2) / f_L2
print(E_N_L2)