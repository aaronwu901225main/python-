# The very first move
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
a=[1, "a", 1., [1,2], (1,2), {'a':3}]
print(a)
y = np.array([[1.,2,3], [4,5,6], [7,8, 9.3]])
print(y)
print(np.array([b'a', b'b', b'cuuuuuuuuiiiiiiiiiiiiiiiiiiii']))
y.shape[0] * y.shape[1] * 8
print(y)
z=np.array([y, 2*y, 3*y, 4*y, 5*y, 6*y])
print(z)
print(z.shape)
x = np.array([1,2,3])
x[0]= 5555555.
print(x)
print(x.shape, x.dtype)
print(y.shape, y.dtype)
x = np.array([10,2,3,4])
# import matplotlib
import matplotlib.pyplot as plt
# plot the graph
x[0]=3
plt.plot(x, 'rv')
plt.show()
from rich import print
print(y)
print({
    "a": 3,
    "b": [1,2,3]
})