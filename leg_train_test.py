#!/usr/bin/env python
import rbdl
import model
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from ilqr.dynamics import constrain
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from ilqr import iLQR, RecedingHorizonController
from ilqr.controller import RecedingHorizonControllerPath
from ilqr.cost import QRCost, PathQRCost, PathQsRCost, PathQRCostMPC
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.dynamics import AutoDiffDynamics, BatchAutoDiffDynamics, FiniteDiffDynamics
import random
import sys
from tqdm import tqdm

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

max_bounds = 8.0
min_bounds = -8.0


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

runner = TPGMMRunner.TPGMMRunner("/home/jack/catkin_ws/src/ambf_walker/Train/gotozero.pickle")

x_path = []
u_path = []

count = 0
N = runner.get_length()
while count < runner.get_length():
    count += 1
    runner.step()
    u_path.append(runner.ddx.flatten().tolist())
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
R = 0.01 * np.eye(dynamics.action_size)
# R[1,1] = 40.0
# R[4,4] = 40.0
#
# R[1,1] = 40.0
# R[4,4] = 40.0

print(R)
#
#
cost = PathQRCost(Q[0], R, x_path=x_path, u_path=u_path)
#
# # Random initial action path.
us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
#
J_hist = []
ilqr = iLQR(dynamics, cost, N-1)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)

R = 0.01 * np.eye(dynamics.action_size)

#
# R[1,1] = 40.0
# R[4,4] = 40.0


cost2 = PathQRCostMPC(Q[0], R, x_path, us)


ilqr2 = iLQR(dynamics, cost2, N-1)


cntrl = RecedingHorizonControllerPath(x0, ilqr2)

# plt.ion()
#
# count = 0
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlim([0, 200])
# ax.set_ylim([-0.10, -0.80])
# x = []
# y = []
# x_follow = []
# y_follow = []
# us3 = []
#
# line1, = ax.plot(x, y, 'r-')
# line2, = ax.plot(x_follow, y_follow, 'b-')
# count = 0
#
# print(sys.getsizeof(cntrl.control(us)))
#
# # for a, b in cntrl.control(us):
# #     print(a)
# #     print(b)
# len_us = len(us)
# print(f"len(us)={len_us}")
#
# # with open('test.npy', 'wb') as f:
# #     np.save(f, us)
# # print(len(us))
# # print('step1')
#
# for xs2, us2 in tqdm(cntrl.control(us)):
#     print(us2[0])
#     us3.append(us2[0])
#     x.append(count)
#     y.append(xs2[0][0])
#     x_follow.append(count)
#     y_follow.append(x_path[count][0])
#     count += 1
#     cntrl.set_state(xs2[1])
#     line1.set_ydata(y)
#     line1.set_xdata(x)
#     line2.set_ydata(y_follow)
#     line2.set_xdata(x_follow)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     us = us[1:]
#     if count == 199:
#         break
# # plt.show()
#
# print(us3)
#
# with open('test2.npy', 'wb') as f:
#     np.save(f, us3)
# print(len(us))

print('step2')
# file = "/home/jack/test.npy"
file = "/home/jack/catkin_ws/src/linearize_model/test2.npy"
with open(file, 'rb') as f:
    us2 = np.load(f)

y = x0
path = []
#plt.figure(2)
for i in range(N-1):
    y = f2(y, us2[i], i)
    path.append(y)

path = np.array(path)

f,  axes = plt.subplots(2, 3, sharex=True)

axes[0, 0].plot(x_path[:, 0], "r")
axes[0, 0].plot(path[:, 0], "g")
axes[0, 1].plot(x_path[:, 1], "r")
axes[0, 1].plot(path[:, 1], "g")
axes[0, 2].plot(x_path[:, 2], "r")
axes[0, 2].plot(path[:, 2], "g")

axes[1, 0].plot(us2[:, 0], "r")
axes[1, 1].plot(us2[:, 1], "r")
axes[1, 2].plot(us2[:, 2], "r")


# _ = ax1.set_title("Trajectory of Hip")
# _ = ax1.set_ylabel("angle")
# _ = ax1.plot(x_path[:, 2], "r")
# _ = ax1.plot(path[:, 2], "g")
# _ = ax1.legend(["Desired", "actually"])
#
# _ = ax2.set_title("Control Signal")
# _ = ax2.plot(us[:, 2], "g")
# _ = ax2.set_ylabel("Torque")
# _ = ax2.legend(["Control sig"])
# _ = ax2.set_label("Time step")


plt.show()
