import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])

b = (a + b)[0:4].tolist()
print(b)
c = [12, 3, 4, 5]
c[2:4]=b[1:3]
print(c)
