#!/usr/bin/env python
import model
import numpy as np
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from ilqr import iLQR
from ilqr.controller import RecedingHorizonControllerPath
from ilqr.cost import PathQRCost, PathQRCostMPC
from ilqr.dynamics import FiniteDiffDynamics, constrain
from tqdm import tqdm
import scipy.io

dt = 0.01  # Discrete time-step in seconds.
tf = 2.0
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

my_model = model.dynamic_model()
J_hist = []


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)

# max_bounds = 10.0
# min_bounds = -10.0


##### 8 and -8
max_bounds = 10.0
min_bounds = -10.0


def f(x, us, i):

    # diff = (max_bounds - min_bounds) / 2.0
    # mean = (max_bounds + min_bounds) / 2.0
    # us = diff * np.tanh(u) + mean
    y = model.runge_integrator(my_model, x, 0.01, us)

    return np.array(y)

def f2(x, u, i):

    y = model.runge_integrator(my_model, x, 0.01, u)

    return np.array(y)
############################################## MODIFY ####################

###### original : return np.array(y)



dynamics = FiniteDiffDynamics(f, 12, 6)

file_name = "/home/jack/backup/leg1.pickle"
#file_name = "/home/jack/catkin_ws/src/ambf_walker/Train/gotozero.pickle"
runner = TPGMMRunner.TPGMMRunner(file_name)

x_path = []
u_path = []

#us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
count = 0
N = runner.get_length()
print(N)
while count < runner.get_length():
    count += 1
    runner.step()
    u_path.append(runner.ddx.flatten().tolist())
    ##### set it to be random no from -5 to 5 (u_p)
    x = runner.x.flatten().tolist() + runner.dx.flatten().tolist()
    x_path.append(x)

u_path = u_path[:-1]
expSigma = runner.get_expSigma()
size = expSigma[0].shape[0]
Q = [np.zeros((size*2, size*2))]*len(expSigma)
for ii in range(len(expSigma)-2, -1, -1):
    Q[ii][:size, :size] = np.linalg.pinv(expSigma[ii])

x0 = x_path[0]
x_path = np.array(x_path)
u_path = np.array(u_path)

# R = 0.05 * np.eye(dynamics.action_size) # +/-10
R = 4.0e-4 * np.eye(dynamics.action_size)

# R[0,0] = 3.0e-3
# R[1,1] = 3.0e-3
# R[2,2] = 3.0e-3
R[3,3] = 7.0e-4
R[4,4] = 7.0e-4
R[5,5] = 7.0e-4
#R[1,1] = 0.00005
# R[4,4] = 5.0e-6
#
# R[1,1] = 40.0
# R[4,4] = 40.0


# print(x_path)

print(R)

# print(u_path)

us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
#
#
cost = PathQRCost(Q[0], R, x_path=x_path, u_path=u_path)

#print(cost)
#
# # Random initial action path.

#
J_hist = []
ilqr = iLQR(dynamics, cost, N-1)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)
# len_us = len(us)
# print(f"len(us)={len_us}")
# print(us)
# print(xs)

#R = 0.01 * np.eye(dynamics.action_size)

#
# R[1,1] = 40.0
# R[4,4] = 40.0
#
# us = np.where(us>max_bounds,max_bounds,us)
# us = np.where(us<min_bounds,min_bounds,us)

# us = constrain(us, min_bounds, max_bounds)
#
# print(us)

# len_us = len(us)
# print(f"len(us)={len_us}")
# print(us)
# print(xs)

#R = 0.01 * np.eye(dynamics.action_size)

#
# R[1,1] = 40.0
# R[4,4] = 40.0

# us = np.where(us>max_bounds,max_bounds,us)
# us = np.where(us<min_bounds,min_bounds,us)

#print(us)

cost2 = PathQRCostMPC(Q[0], R, x_path, us)


ilqr2 = iLQR(dynamics, cost2, N-1)


cntrl = RecedingHorizonControllerPath(x0, ilqr2)

# print(J_hist)
#
# ###### if want to smooth J_hist
#
plt.figure()

plt.plot(J_hist)

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid()

plt.show()

J_arr = np.array(J_hist)

scipy.io.savemat('cost.mat',mdict={'arr':J_arr})




