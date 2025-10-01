import numpy as np

a = np.matrix([[1, 2, 3]
            , [4, 5, 6], 
                [7, 8, 9]])
b = np.matrix([[9, 8, 7]
            , [6, 5, 4], 
                [3, 2, 1]])
c = a + b
print(c)

d = a * b
print(d)

e = a - b
print(e)

f = a.T
print(f)

g = np.linalg.inv(a)
print(g)

h = np.linalg.det(a)
print(h)

i = np.trace(a)
print(i)

j = np.linalg.eig(a)
print(j)

k = np.linalg.svd(a)
print(k)

l = np.linalg.solve(a, b)
print(l)