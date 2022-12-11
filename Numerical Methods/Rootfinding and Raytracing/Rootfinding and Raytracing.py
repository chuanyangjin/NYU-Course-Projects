# Question 1
import numpy as np
from scipy import optimize

tol = 1e-13

# calculate the derivative of a polynomial p
def derivative(p):
    res = []
    for i in range(1,len(p)):
        res.append(p[i]*i)
    return np.array(res)

# calculate the value of a polynomial p at a
def value(p, a):
    res = 0
    for i in range(len(p)):
        res += p[i] * a**i
    return res

# given two polynomials p1, p2, calculate the quotient result of polynomial long division (without the remainder)
def divide(p, q):
    p = p.copy()
    q = q.copy()
    p = p[::-1]
    q = q[::-1]
    res = []
    while len(p) >= len(q):
        n = p[0] / q[0]
        for i in range(0, len(q)):
            p[i] -= q[i] * n
        p = p[1:]
        res.append(n)
    return np.array(res)[::-1]

# given two polynomials p, q, calculate their product
def multiply(p, q):
    res = [0] * (len(p)+len(q)-1)
    for i in range(len(p)):
        for j in range(len(q)):
            res[i+j] += p[i] * q[j]
    return np.array(res)

# given two polynomials p, q, calculate p minus q
def minus(p, q):
    res = [0] * max(len(p),len(q))
    for i in range(len(p)):
        res[i] += p[i]
    for i in range(len(q)):
        res[i] -= q[i]

    # get rid of the leading 0's
    i = len(res)-1
    while abs(res[i]) < tol and i>0:
        res = res[:-1]
        i -= 1
    return np.array(res)

# count the number of real roots of p in the interval (a,b)
def countroots(p, a, b):
    # construct a Sturm Chain
    chain = [p]                             # p0 = p
    if len(p) > 1:
        chain.append(derivative(p))         # p1 = p'
    while len(chain[-1]) > 1:               # calculate the sequence until pn is constant
        qn = divide(chain[-2], chain[-1])
        pn = minus(multiply(qn, chain[-1]), chain[-2])
        chain.append(pn)
    
    # count the number of sign changes in p0(a),  ..., pn(a) and p0(b),  ..., pn(b)
    # the differents between them determines the number of real roots
    count1, count2 = 0, 0
    for i in range(len(chain)-1):
        if (value(chain[i],a)>=0 and value(chain[i+1],a)<0) or (value(chain[i],a)<0 and value(chain[i+1],a)>=0):
            count1 += 1
        if (value(chain[i],b)>=0 and value(chain[i+1],b)<0) or (value(chain[i],b)<0 and value(chain[i+1],b)>=0):
            count2 += 1
    return abs(count1 - count2)

# find the real roots of p in the interval [a,b]
def findroots(p, a, b):
    # check the open interval (a,b)
    roots = findroots2(p, a, b)

    # check the bounds x=a and x=b
    if value(p,a) == 0:
        roots = np.append(roots,a)
    if value(p,b) == 0:
        roots = np.append(roots,b)
    return sorted(np.unique(roots))

# find the real roots of p in the interval (a,b)
def findroots2(p, a, b):
    n = countroots(p, a, b)     # count the number of real roots of p in the interval (a,b)
    if abs(a-b) < tol:          # the inverval (a,b) is too small
        roots = [a] * n  
    elif n == 0:                # no roots in (a,b)
        roots = []
    elif n == 1:                # exact one root in (a,b), use brentq to solve it
        def f(x):
            return value(p,x)
        x = optimize.brentq(f, a, b)
        roots = [x]
    else:                       # at least two roots, divide and conquer
        mid = (a+b) / 2
        left = findroots2(p, a, mid)
        right = findroots2(p, mid, b)
        roots = np.append(left,right)
    return np.array(sorted(roots))

# p(x) = (x - 0)*(x - 0.25)*(x - 0.5)*(x - 0.75)*(x - 1.0)
p = np.array([0.0, 0.09375, -0.78125, 2.1875, -2.5, 1.0])
roots_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# this choice of (a, b) should be easy if you do the bisection right
a, b = -0.1, 1.2
roots = findroots(p, a, b)
print(roots)
# check relative error... be careful how you do this!
assert (abs(roots - roots_true) < tol*roots_true + tol).all()

# this choice will trigger corner cases
a, b = 0.0, 1.0
roots = findroots(p, a, b)
print(roots)
assert (abs(roots - roots_true) < tol*roots_true + tol).all()



# Question 2
# given a ray r(t) = r + td with origin r and direction d, and a Goursat’s surface with parameters a,b,c
# return the intersection of the ray and the surface
def getintersection(r,d,a,b,c):
    # r: origin, d: direction
    x = [r[0], d[0]] 
    y = [r[1], d[1]] 
    z = [r[2], d[2]]
    first = np.polynomial.polynomial.polypow(x,4) + np.polynomial.polynomial.polypow(y,4) + np.polynomial.polynomial.polypow(z,4)
    second =  a * (np.polynomial.polynomial.polypow(x,2) + np.polynomial.polynomial.polypow(y,2) + np.polynomial.polynomial.polypow(z,2))**2
    third = b * (np.polynomial.polynomial.polypow(x,2) + np.polynomial.polynomial.polypow(y,2) + np.polynomial.polynomial.polypow(z,2))
    fourth = [c]
    temp1 = np.polynomial.polynomial.polyadd(first, second)
    temp2 = np.polynomial.polynomial.polyadd(third, fourth)
    p = np.polynomial.polynomial.polyadd(temp1, temp2)
    root = findroots(p, -1, 1)

    if len(root) == 0:     # if no roots, return None
        return None
    else:                  # if has root, get the first intersection
        root = root[0]
        intersection = [0] * 3
        for i in range(3):
            intersection[i] = r[i] + root * d[i]
        return intersection

# normalize a vector
def normalize(vector):
    return vector / np.linalg.norm(vector)

# given a Goursat’s surface with parameters a,b,c, and a point on that surface
# return the surface normal at that point
def surface_normal(point,a,b,c):
    x,y,z = point[0],point[1],point[2]
    x_normal = (4*a+4) * x**3 + (4*a*y**2 + 4*a*z**2 + 2*b) * x
    y_normal = (4*a+4) * y**3 + (4*a*x**2 + 4*a*z**2 + 2*b) * y
    z_normal = (4*a+4) * z**3 + (4*a*x**2 + 4*a*y**2 + 2*b) * z
    return normalize([x_normal, y_normal, z_normal])

# given a ray r(t) = r + td with origin r and direction d, and a Goursat’s surface with parameters a,b,c
# return the color based on the angle that the ray makes with the surface
def color(r,d,a,b,c):
    # get the intersection of ray and surface
    intersection = getintersection(r,d,a,b,c)
    if intersection == None:
        return 0

    # get the unit direction
    d = normalize(d)

    # get the surface normal at the intersection
    normal_vector = surface_normal(intersection,a,b,c)

    # get the color parameter (- n·d)
    product = 0
    for i in range(3):
        product -= d[i] * normal_vector[i]
    return product

# an example
# Goursat’s surface: a, b, c = 1, 0, -1
#                    p(x,y,z) = x**4 + y**4 + z**4 + (x**2 + y**2 + z**2)**2 - 1 = 0
# direction (0,0,1)
# initial ray positions are on a plane -2 <= x <= 2, -2 <= y <= 2, z = 1
a, b, c = 1, 0, -1
d = [0, 0, 1]
arr = [[0 for i in range(256)] for j in range(256)]
for i in range(256):
    for j in range(256):
        r = [-2+i/64,-2+j/64,1]
        print(r, end = " ")
        arr[i][j] = color(r,d,a,b,c)
        print(arr[i][j])

# save the image as 'prog1.jpg'
import matplotlib.pyplot as plt
plt.imsave('prog1.jpg',arr)