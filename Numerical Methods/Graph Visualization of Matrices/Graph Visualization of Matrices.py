import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg

tol = 1e-13

# Problem 1
def power_method(A):
    u = np.ones(len(A))
    u = u / np.linalg.norm(u)
    lam = u.T@A@u
    lams = []
    while True:
        u = A@u
        u = u / np.linalg.norm(u)
        lam_new = u.T@A@u
        if (abs(lam_new - lam) < tol):
            break
        lams.append(lam)
        lam = lam_new
    return lam, u, lams

def inverse_power_method(A):
    n = len(A)
    u = np.ones(n)

    P, L, U = scipy.linalg.lu(A)
    L_inverse = scipy.linalg.solve_triangular(L, np.identity(n),lower=True)
    #print(L_inverse)
    U_inverse = scipy.linalg.solve_triangular(U, np.identity(n))
    A_inv = U_inverse@L_inverse@P
    #print(A_inv@A)

    lam = u.T@A@u
    lams = []
    while True:
        #u = np.linalg.inv(A)@u   # don't use np.linalg.inv(); instead, let A = LU, then A^{-1} = U^{-1}L^{-1}

        u = A_inv@u
        u = u / np.linalg.norm(u)
        lam_new = u.T@A@u
        if (abs(lam_new - lam) < tol):
            break
        lams.append(lam)
        lam = lam_new
    return lam, u, lams

n = 4
Q = np.random.randn(n, n) # sample a matrix with iid N(0, 1) entries
Q = np.linalg.qr(Q)[0] # the Q factor in its QR decomposition is an
                       # orthogonal matrix sampled uniformly from
                       # the orthgonal group O(n)
Lam1 = np.diag([30, 3, 1.1, 1]) # set the diagonal matrix of eigenvalues manually
A1 = Q@Lam1@Q.T # compute A from Q and Lam
Lam2 = np.diag([8, 4, 2, 1]) # set the diagonal matrix of eigenvalues manually
A2 = Q@Lam2@Q.T # compute A from Q and Lam
Lam1 = np.diag([5.5, 5, 5, 0.5]) # set the diagonal matrix of eigenvalues manually
A3 = Q@Lam1@Q.T # compute A from Q and Lam

lam1, u1, lams1 = power_method(A1)
lam2, u2, lams2 = power_method(A2)
lam3, u3, lams3 = power_method(A3)
print(lam1, lam2, lam3)

plt.semilogy(abs(lams1 - lam1), label = "lam1:lam2 = 10:1")
plt.semilogy(abs(lams2 - lam2), label = "lam1:lam2 = 2:1")
plt.semilogy(abs(lams3 - lam3), label = "lam1:lam2 = 1.1:1")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.title("Convergence of power method")
plt.legend()
plt.show()

lam1, u1, lams1 = inverse_power_method(A1)
lam2, u2, lams2 = inverse_power_method(A2)
lam3, u3, lams3 = inverse_power_method(A3)
print(lam1, lam2, lam3)

plt.semilogy(abs(lams1 - lam1), label = "lam_{n-1} : lam_n = 1.1:1")
plt.semilogy(abs(lams2 - lam2), label = "lam_{n-1} : lam_n = 10:1")
plt.semilogy(abs(lams3 - lam3), label = "lam_{n-1} : lam_n = 25:1")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.title("Convergence of inverse power method")
plt.legend()
plt.show()


# Problem 2
def shifted_power_method(A):
    m,n = A.shape
    u = np.ones(len(A))
    u = u / np.linalg.norm(u)
    temp = A
    lams = []
    i = 0
    while True:
        i=i+1
        shift = 0
        
        if i > 10:
            shift = (np.trace(A) - A@u@u) / n
        A = A - shift * np.identity(len(A))
        lam = u.T@A@u + shift

        Au = A@u
        u = Au / np.linalg.norm(Au)
        lam_new = u.T@A@u + shift
        if (abs(lam_new - lam) < tol):
            break
        lams.append(lam)
        lam = lam_new
        A = temp
    return lam, u, lams

def inverse_shifted_power_method(A):
    m,n = A.shape
    u = np.ones(len(A))
    u = u / np.linalg.norm(u)
    i = 0
    I = np.eye(n)
    lam = u.T@A@u
    lams = []
    temp = A
    shift = 0
    while True:
        # u = np.linalg.inv(A)@u   # don't use np.linalg.inv(); instead, let A = LU, then A^{-1} = U^{-1}L^{-1}
        i += 1
        if(i > 1000):
            inverse_power_method(A)

        A = A - shift * I
        P, L, U = scipy.linalg.lu(A)
        v = np.dot(P.T,u)
        v= scipy.linalg.solve_triangular(L,v,lower=True)
        v= scipy.linalg.solve_triangular(U,v)
        L_inverse = scipy.linalg.solve_triangular(L, np.identity(n),lower=True)
        U_inverse = scipy.linalg.solve_triangular(U, np.identity(n))
        A_inv = U_inverse@L_inverse@P
        
        if i > 10:
            shift = 1 / np.linalg.norm(v)
        lam = u.T@A@u

        Au = A_inv@u
        u = Au / np.linalg.norm(Au)
        lam_new = u.T@A@u
        abs(lam_new - lam)
        if (abs(lam_new - lam) < tol):
            break
        lams.append(lam)
        lam = lam_new
        A = temp
    lams.append(lam_new)
    return lam, u, lams

# Lam4 = np.diag([5.5, 5, 5, 0.5]) # set the diagonal matrix of eigenvalues manually
Lam4 = np.diag([5.5, 5, 1.5, 1]) # set the diagonal matrix of eigenvalues manually
A4 = Q@Lam4@Q.T # compute A from Q and Lam

lam, u, lams = power_method(A4)
plt.semilogy(abs(lams - lam), label = "power method")
lam_shifted, u_shifted, lams_shifted = shifted_power_method(A4)
print(lam_shifted)
plt.semilogy(abs(lams_shifted - lam_shifted), label = "shifted power method")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.title("Convergence of shifted power method")
plt.legend()
plt.show()

lam, u, lams = inverse_power_method(A4)
plt.semilogy(abs(lams - lam), label = "inverse power method")
lam_shifted, u_shifted, lams_shifted = inverse_shifted_power_method(A4)
print(lam_shifted)
plt.semilogy(abs(lams_shifted - lam_shifted), label = "inverse shifted power method")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.title("Convergence of shifted inverse power method")
plt.legend()
plt.show()


# Problem 3
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
def compute_A(N):
    # A = np.zeros(((N-1)**2, (N-1)**2))
    row = []
    col = []
    data = []
    for i in range((N-1)**2):
        # A[i,i] = 4
        row.append(i)
        col.append(i)
        data.append(4.0)
        if i >= N-1:
            # A[i, i - (N-1)] = -1
            row.append(i)
            col.append(i - (N-1))
            data.append(-1)
        if i < (N-2) * (N-1):
            # A[i, i + (N-1)] = -1
            row.append(i)
            col.append(i + (N-1))
            data.append(-1)
        if i % (N-1) != 0:
            # A[i, i-1] = -1
            row.append(i)
            col.append(i - 1)
            data.append(-1)
        if i % (N-1) != N-2:
            # A[i, i+1] = -1
            row.append(i)
            col.append(i + 1)
            data.append(-1)
    A = csr_matrix((data, (row, col)), shape=((N-1)**2, (N-1)**2))
    return A

A = compute_A(64)
# eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A, k = 16, sigma = 0)
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A, k = 16, which='SM')

for k in range(16):
    e = eigenvectors[:, k]
    graph = np.identity(63)
    for i in range(63):
        for j in range(63):
            graph[i][j] = (e[(i-1)*63 + j])
    plt.subplot(4, 4, k+1)
    im = plt.imshow(graph)
    plt.colorbar(im)
plt.show()

plt.plot(eigenvalues)
plt.show()

# Problem 4a
import csv
d = {}
with open('Example Data.csv', mode='r') as inp:
    reader = csv.reader(inp)
    d = {rows[0]:rows[1:] for rows in reader}
for key in d:
    d[key] = list(filter(lambda val: val !=  "", d[key]))

row = []
col = []
data = []
for i in range(500):
    # L[i,i] = deg(v_i)
    row.append(i)
    col.append(i)
    data.append(len(d[str(i)]))
    for j in d[str(i)]:
        # L[i,j] = -1
        row.append(i)
        col.append(float(j))
        data.append(-1.0)
L = csr_matrix((data, (row, col)), shape=(500,500))
print(L.toarray())

eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k = 3, sigma = 0)
print(eigenvalues)      # the first one is smaller than tol
u1 = eigenvectors[:, 1]
u2 = eigenvectors[:, 2]

plt.plot(u1, u2)
plt.show()

# An idea that doesn't work
# graph = np.identity(500)
# for i in range(500):
#     for j in range(500):
#         graph[i][j] = u1[i] * u2[j]
# im = plt.imshow(graph)
# plt.colorbar(im)
# plt.show()

# Problem b
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k = 4, sigma = 0)
print(eigenvalues)      # the first one is smaller than tol
u1 = eigenvectors[:, 1]
u2 = eigenvectors[:, 2]
u3 = eigenvectors[:, 3]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(u1, u2, u3)
plt.show()

# Problem 5
row = []
col = []
data = []
for i in range(500):
    for j in d[str(i)]:
        # L[i,j] = -1
        row.append(i)
        col.append(float(j))
        data.append(1.0)
A = csr_matrix((data, (row, col)), shape=(500,500))
print(A.toarray())

S, V, D = np.linalg.svd(A.toarray(), full_matrices = False)
plt.semilogy(V, label = "The Decay of Singular Values")
plt.show()
# decay slowly at first, then faster

im = plt.imshow(A.toarray(), vmax = 0.5, vmin = -0.5)
plt.colorbar(im)
plt.show()

k = 500
A_k = S @ (V[..., None] * D)
im = plt.imshow(A_k, vmax = 0.5, vmin = -0.5)
plt.colorbar(im, label="k = 500")
plt.show()
print(S.shape)
print(V.shape)
print(V[..., None].shape)
print(D.shape)

k = 400
A_k = S[:,0:k] @ (V[0:k, None] * D[0:k])
im = plt.imshow(A_k, vmax = 0.5, vmin = -0.5)
plt.colorbar(im, label="k = 400")
plt.show()

k = 100
A_k = S[:,0:k] @ (V[0:k, None] * D[0:k])
im = plt.imshow(A_k, vmax = 0.5, vmin = -0.5)
plt.colorbar(im, label="k = 100")
plt.show()

k = 20
A_k = S[:,0:k] @ (V[0:k, None] * D[0:k])
im = plt.imshow(A_k, vmax = 0.5, vmin = -0.5)
plt.colorbar(im, label="k = 20")
plt.show()

k = 5
A_k = S[:,0:k] @ (V[0:k, None] * D[0:k])
im = plt.imshow(A_k, vmax = 0.5, vmin = -0.5)
plt.colorbar(im, label="k = 5")
plt.show()

# When k = p, the matrix we reconstruct is exactly the same as the original one.
# While k becomes smaller, the reconstruct matrices will lose more and more details.
# The amount of information remained is related to the scale of the first k eigenvalues. 