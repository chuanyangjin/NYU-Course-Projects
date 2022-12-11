from cmath import inf
import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt

# Problem 1
def compute_A(N):
    A = np.zeros(((N-1)**2, (N-1)**2))
    for i in range((N-1)**2):
        A[i,i] = 4
        if i >= N-1:
            A[i, i - (N-1)] = -1
        if i < (N-2) * (N-1):
            A[i, i + (N-1)] = -1
        if i % (N-1) != 0:
            A[i, i-1] = -1
        if i % (N-1) != N-2:
            A[i, i+1] = -1
    return A

A = compute_A(10)
print(A.shape)
im = plt.imshow(A,cmap="seismic",vmin=-4,vmax=4)
plt.colorbar(im)
plt.show()

# Problem 2
def lu(A):
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy()
    P = np.eye(n)
    for j in range(n):
        maxi = -inf
        for i in range(j,n):
            if U[i,j] > maxi:
                maxi = U[i,j]
                k = i
        U[j], U[k] = U[k], U[j]
        P[j], P[k] = P[k], P[j]

        L[j,j] = 1
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i] -= L[i,j] * U[j]
    return L, U, P

epsilon = 0.00001
L,U,P = lu(A)
P0,L0,U0 = scipy.linalg.lu(A)
assert (P0 == P).all()
assert (abs(L0 - L) < epsilon).all()
assert (abs(U0 - U) < epsilon).all()

# Problem 3
def find_lower_bandwidth(L):
    ans = 0
    n = len(L)
    for x in range(1, n):
        xtemp = x
        ytemp = 0
        while(xtemp != n-1):
            if(L[xtemp,ytemp] == 0):
                xtemp = xtemp + 1
                ytemp = ytemp + 1
            elif(L[xtemp,ytemp] != 0):
                ans = ans + 1
                xtemp = n-1
    return ans

x = [10, 20, 30, 40, 50]
nonzeros = []
lower_bandwidths = []
for N in x:
    L,U,P = lu(compute_A(N))
    im = plt.imshow(L,cmap="seismic",vmin=-4,vmax=4)
    plt.colorbar(im)
    plt.show()

    nonzero = np.count_nonzero(L)
    nonzeros.append(nonzero)
    lower_bandwidth = find_lower_bandwidth(L)
    lower_bandwidths.append(lower_bandwidth)

plt.plot(x,nonzeros, 'o')
plt.xlabel("N")
plt.ylabel("Nonzeros")
plt.title("nonzeros")
plt.show()

plt.plot(x,lower_bandwidths, 'o')
plt.xlabel("N")
plt.ylabel("Lower Bandwidths")
plt.title("lower_bandwidths")
plt.show()

# Problem 4
def fsolve(L, b):
    n = len(L)
    b[0] = b[0] / L[0,0]
    for i in range(1,n):
        b[i] = (b[i] - np.dot(L[i,0:i-2], b[0:i-2])) / L[i,i]
    return b

def bsolve(U, b):
    n = len(U)
    b[n-1] = b[n-1] / U[n-1,n-1]
    for i in range(n-2,-1,-1):
        b[i] = (b[i] - np.dot(U[i,i+1:n], b[i+1:n])) / U[i,i]
    return b

def plot(i,j):
    e = np.zeros((50-1)**2)
    e[(50-1) * (i-1) + (j-1)] = 1

    L,U,P = lu(A)
    UX = fsolve(L, np.dot(P,e))
    X = bsolve(U, UX)

    a,b,c = [],[],[]
    for i in range(1,50):
        for j in range(1,50):
            a.append(i)
            b.append(j)
            c.append(X[(50-1) * (i-1) + (j-1)])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(a,b,c)

A = compute_A(50)
plot(1,1)
plt.show()
plot(1,2)
plt.show()
plot(25,25)
plt.show()
plot(10,30)
plt.show()