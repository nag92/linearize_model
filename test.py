import numpy as np
import time

a = np.array([ [1.0,2.0,3.0,],[4,5,6],[7,8,9] ])
start_time = time.time()
temp = np.zeros(a.shape)
temp[:-1] = a[1:]
temp[-1] = a[-1]
a = temp
t1 = time.time() - start_time
print(temp)

a = np.array([ [1.0,2.0,3.0,],[4,5,6],[7,8,9] ])
start_time = time.time()
a = a[1:]
a = np.append(a, [a[-1].tolist()], axis=0)
t2 = time.time() - start_time
print(a)
print(t1)
print(t2)