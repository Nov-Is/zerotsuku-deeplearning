>>> import numpy as np
>>> x = np.array([1.0, 2.0, 3.0])
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
>>> y = np.array([4.0, 5.0, 6.0])
>>> x + y
array([5., 7., 9.])
>>> y = np.array([2.0, 4.0, 6.0])
>>> x + y
array([3., 6., 9.])
>>> x - y
array([-1., -2., -3.])
>>> x * y
array([ 2.,  8., 18.])
>>> x / y
array([0.5, 0.5, 0.5])
>>> A = np.array([[1, 2], [3, 4]])
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int64')
>>> B = np.array([[3, 0],[0, 6]])
>>> A + B
array([[ 4,  2],
       [ 3, 10]])
>>> A * B
array([[ 3,  0],
       [ 0, 24]])
>>> print(A)
[[1 2]
 [3 4]]
>>> print(A * 10)
[[10 20]
 [30 40]]
>>> A = np.array([[1, 2],[3, 4 ]])
>>> B = np.array([10, 20])
>>> A * B
array([[10, 40],
       [30, 80]])
>>> X = np.array([[51, 55], [14, 19], [0, 4]])
>>> print(X)
[[51 55]
 [14 19]
 [ 0  4]]
>>> X[0]
array([51, 55])
>>> X[0][1]
np.int64(55)
>>> for row in X:
...     print(row)
...     
[51 55]
[14 19]
[0 4]
>>> X = X.flatten()
>>> print(X)
[51 55 14 19  0  4]
>>> X[np.array([0, 2, 4])]
array([51, 14,  0])
>>> X > 15
array([ True,  True, False,  True, False, False])
>>> X[X > 15]
array([51, 55, 19])
