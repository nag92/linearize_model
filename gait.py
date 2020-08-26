#!/usr/bin/env python
# coding: utf-8

# # Gait generation
#
# The state and control vectors $\textbf{x}$ and $\textbf{u}$ are defined as follows:
#
# $$
# \begin{equation*}
# \textbf{X} = \begin{bmatrix}
#     x & y & z & \dot{x} &\dot{y} &\dot{z}
#     \end{bmatrix}^T
# \end{equation*}
# $$
#
# $$
# \begin{equation*}
#     \textbf{u} = \begin{bmatrix}
#     x_z & y_z & F_z
#     \end{bmatrix}^T
# \end{equation*}
# $$
#
#
# The dynamic equation for the system will be
#
# $$
# \begin{equation*}
#     \begin{bmatrix}
#     \dot{X}
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#     \dot{x} \\
#     \dot{y} \\
#     \dot{z} \\
#     \frac{(x-x_z)F_z}{mz} \\
#     \frac{(y-y_z)F_z}{mz} \\
#     \frac{F_z}{m}-g
#     \end{bmatrix}
# \end{equation*}
# $$
#
#
# where $\textbf{m}$ and $\textbf{g}$ are mass and the acceleration of gravity. In the following case, I will set $\textbf{m}$ to be 1.
#
#
#
#
#

# In[1]:



# In[2]:


import matplotlib
# In[3]:


import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

# In[4]:


from ilqr import iLQR
from ilqr.cost import PathQRCost
from ilqr.dynamics import tensor_constrain
from ilqr.dynamics import constrain
from ilqr.dynamics import AutoDiffDynamics


# In[5]:

J_hist = []
def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)


# In[8]:


x_inputs = [
    T.dscalar("x"),
    T.dscalar("y"),
    T.dscalar("z"),
    T.dscalar("x_dot"),
    T.dscalar("y_dot"),
    T.dscalar("z_dot"),
]

x_inputs_dot=[
    T.dscalar("xrk"),
    T.dscalar("yrk"),
    T.dscalar("zrk"),
    T.dscalar("x_dotrk"),
    T.dscalar("y_dotrk"),
    T.dscalar("z_dotrk"),
]

u_inputs = [
    T.dscalar("x_zmp"),
    T.dscalar("y_zmp"),
    T.dscalar("F_z"),
]

t = T.dscalar("t")
dt = 0.01
g = 9.81
min_bounds = np.array([0.0, 0.0, 0.0])
max_bounds = np.array([300.0, 50.0, 100])
u_constrained_inputs = tensor_constrain(u_inputs, min_bounds, max_bounds)
#nonlinear dynamics.


def nonldyn(x_inputs, u_inputs):
    x_inputs_dot[0]=x_inputs[3],
    x_inputs_dot[1]=x_inputs[4],
    x_inputs_dot[2]=x_inputs[5],
    x_inputs_dot[3]=(x_inputs[0]-u_inputs[0])*u_inputs[2]/x_inputs[2],
    x_inputs_dot[4]=(x_inputs[1]-u_inputs[1])*u_inputs[2]/x_inputs[2],
    x_inputs_dot[5]=u_inputs[2]-g,
    #return x_inputs_dot
    return T.stack([
        x_inputs_dot[0],
        x_inputs_dot[1],
        x_inputs_dot[2],
        x_inputs_dot[3],
        x_inputs_dot[4],
        x_inputs_dot[5], ] )



f = x_inputs+(dt/6)*(nonldyn(x_inputs,u_inputs)+2*nonldyn(x_inputs+.5*dt*nonldyn(x_inputs,u_inputs),u_inputs)+2*nonldyn(x_inputs+.5*dt*nonldyn(x_inputs+.5*dt*nonldyn(x_inputs,u_inputs),u_inputs),u_inputs)+nonldyn(x_inputs+dt*nonldyn(x_inputs+.5*dt*nonldyn(x_inputs+.5*dt*nonldyn(x_inputs,u_inputs),u_inputs),u_inputs)))

f = T.stack([
    x_inputs[0] + x_inputs[3] * dt,
    x_inputs[1] + x_inputs[4] * dt,
    x_inputs[2] + x_inputs[5] * dt,
    x_inputs[3] + ((x_inputs[0]-u_constrained_inputs[0])*u_constrained_inputs[2]/x_inputs[2]) * dt,
    x_inputs[4] + ((x_inputs[1]-u_constrained_inputs[1])*u_constrained_inputs[2]/x_inputs[2] )* dt,
    x_inputs[5] + (u_constrained_inputs[2]-g) * dt,
])

dynamics = AutoDiffDynamics(f, x_inputs, u_inputs, t)

nos=50
xindex=[]
for i in range (1,nos+2):
    xindex.append(i)
uindex=[]
for i in range (1,nos+1):
    uindex.append(i)

#xs good, us not good, xs2 not good as well
#Q = np.diag((1000,1000,2080,0.001,.001,.001))
#R = np.diag((0.1,0.1,0.001))

#relatively good, but with oscillation
#Q = np.diag((15000,15000,25000,0.0001,.0001,.00001))
#R = np.diag((1200,1200,0.00008))

Q = np.diag((15000,15000,25000,0.0001,.0001,.00001))
R = np.diag((8000,8000,0.00008))

nos=50
xindex=[]
for i in range (1,nos+2):
    xindex.append(i)
uindex=[]
for i in range (1,nos+1):
    uindex.append(i)

#xs good, us not good, xs2 not good as well
#Q = np.diag((1000,1000,2080,0.001,.001,.001))
#R = np.diag((0.1,0.1,0.001))

#relatively good, but with oscillation
#Q = np.diag((15000,15000,25000,0.0001,.0001,.00001))
#R = np.diag((1200,1200,0.00008))

Q = np.diag((15000,15000,25000,0.0001,.0001,.00001))
R = np.diag((8000,8000,0.00008))


a = np.array([10,11.904,13.626,15.26,16.897,18.63,20.551,22.752,25.318,28.223,31.351,34.587,37.813,40.912,43.768,46.263,48.33,50.015,51.378,52.482,53.388,54.156,54.85,55.53,56.258,57.095,58.103,59.343,60.877,62.767,65.072,67.779,70.774,73.932,77.131,80.249,83.161,85.746,87.906,89.671,91.1,92.256,93.2,93.994,94.699,95.377,96.089,96.896,97.861,99.044,5,5.0164,5.1312,5.4429,6.0498,7.0503,8.5429,10.626,13.388,16.759,20.545,24.55,28.577,32.429,35.909,38.821,41.049,42.664,43.765,44.451,44.82,44.97,45,44.992,44.908,44.648,44.116,43.211,41.836,39.893,37.285,34.033,30.322,26.349,22.31,18.402,14.822,11.767,9.39,7.6471,6.44,5.6703,5.2396,5.0494,5.0014,5.0028,5.0617,5.2738,5.7373,6.5507,90.531,89.044,87.738,86.63,85.726,85.063,84.737,84.92,85.849,87.645,90.055,92.48,94.223,94.842,94.348,93.138,91.71,90.369,89.237,88.331,87.621,87.053,86.575,86.14,85.718,85.301,84.904,84.582,84.446,84.68,85.533,87.19,89.515,92.005,93.979,94.91,94.692,93.641,92.262,90.911,89.746,88.803,88.059,87.467,86.975,86.533,86.109,85.687,85.276,84.918
])
comref=np.reshape(a,(nos,3),order='F')
x1t3=comref.T
x4t6=np.array([[2.025,1.7978,1.6628,1.6201,1.6695,1.8118,2.0441,2.377,2.7496,3.0372,3.2013,3.2504,3.1819,2.9973,2.6934,2.2825,1.8648,1.514,1.2232,0.99433,0.82687,0.72094,0.67652,0.69361,0.7722,0.91232,1.1139,1.3773,1.7008,2.0912,2.5183,2.8726,3.0966,3.1995,3.1789,3.0353,2.7691,2.3784,1.9525,1.5866,1.2825,1.0399,0.85877,0.73917,0.68107,0.68449,0.74942,0.87586,1.0638,1.3133],[-1.4572e-16,0.04921,0.19682,0.4429,0.78718,1.2307,1.7695,2.4184,3.0916,3.6129,3.9281,4.0489,3.972,3.6996,3.2261,2.5722,1.9036,1.3424,0.87714,0.51086,0.24329,0.072408,0.0065415,-0.031884,-0.15495,-0.37987,-0.70213,-1.1237,-1.6412,-2.2659,-2.9493,-3.5161,-3.8745,-4.0392,-4.0063,-3.7765,-3.3506,-2.7254,-2.0441,-1.4585,-0.97206,-0.58371,-0.29437,-0.1014,-0.014453,0.01935,0.11795,0.32168,0.62196,1.0214],[-1.5717,-1.3995,-1.2086,-1.0082,-0.79343,-0.51654,-0.10808,0.51922,1.3688,2.1784,2.5353,2.1873,1.2188,0.023378,-0.93632,-1.39,-1.4177,-1.2469,-1.0164,-0.80034,-0.63012,-0.514,-0.45012,-0.42536,-0.41966,-0.41234,-0.37292,-0.25356,0.011546,0.50124,1.2454,2.0465,2.5148,2.3418,1.5088,0.336,-0.71229,-1.2928,-1.4061,-1.2723,-1.0542,-0.83662,-0.65912,-0.53365,-0.46009,-0.42875,-0.42198,-0.42004,-0.39552,-0.30494]])

xinitial_trans=np.array([10,5,91,100,100,100])
xinitial=xinitial_trans.reshape(-1, 1)
x_T=np.r_[x1t3,x4t6]
x_path=np.c_[xinitial,x_T].T
upath3=np.full((nos,1),10)
upath1t2=np.c_[x1t3[0,:],x1t3[1,:]]
u_path= np.c_[upath1t2,upath3]

cost = PathQRCost(Q, R, x_path, u_path=u_path)


N = nos  # Number of time steps in trajectory.
x0 = np.array([0, 0, 90.5305, 10, 10, 10])  # Initial state.

# Random initial action path.
us_init = np.random.uniform(-100, 100, (N, dynamics.action_size))
#us_init=u_path

J_hist = []
ilqr = iLQR(dynamics, cost, N)
xs,us_unconstrained = ilqr.fit(x0, us_init,n_iterations=10000,on_iteration=on_iteration)

plt.plot(xindex,xs[:,0])
# plt.plot(xindex,xs[:,1])
# plt.plot(xindex,xs[:,2])

plt.show()