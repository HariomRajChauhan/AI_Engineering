import numpy as np


# basic array operations

# a = np.array([1, 2, 3, 4, 5])
# b = np.array([6, 7, 8, 9, 10])
# c = a + b
# print(c)
# d = a * b
# print(d)
# e = a - b
# print(e)
# f = a / b
# print(f)
# g = a ** 2
# print(g)
# h = np.sqrt(a)
# print(h)
# i = np.log(a)
# print(i)
# j = np.exp(a)
# print(j)
# k = np.sin(a)
# print(k)
# l = np.cos(a)
# print(l)
# m = np.tan(a)
# print(m)
# n = np.sum(a)
# print(n)
# o = np.mean(a)
# print(o)
# p = np.median(a)
# print(p)
# q = np.std(a)
# print(q)

###### Broadcasting in numpy

# using arange and shape to create arrays
# array1 = np.arange(1,10)
# print(array1)
# array1.shape = (3,3)
# print(array1)
# # allows numpy to perform operations on arrays of different shapes
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([[1], [2], [3]])
# print(a.shape, b.shape)
# c = a + b
# print(c)
# # allows numpy to perform operations on arrays of different shapes
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([[1], [2], [3]])
# c = a + b
# print(c)
# # allows numpy to perform operations on arrays of different shapes
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([[1, 2, 3]])
# c = a + b
# print(c)


# # for zeros and ones

# array_zeros = np.zeros(10).reshape(2,5)
# print(array_zeros)

# array_ones = np.ones(10).reshape(5,2)
# print(array_ones)

