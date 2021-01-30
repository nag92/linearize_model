#!/usr/bin/env python
import rbdl
import model
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from ilqr.dynamics import constrain

from ilqr import iLQR, RecedingHorizonController
from ilqr.cost import QRCost, PathQRCost, PathQsRCost

from ilqr.dynamics import AutoDiffDynamics, BatchAutoDiffDynamics, FiniteDiffDynamics


dt = 0.01  # Discrete time-step in seconds.
tf = 2.0
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.
from ilqr.dynamics import FiniteDiffDynamics
my_model = model.dynamic_model()
J_hist = []
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner


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



dynamics = FiniteDiffDynamics(f, 12, 6)

runner = TPGMMRunner.TPGMMRunner("/home/nathanielgoldfarb/catkin_ws/src/ambf_walker/Train/gotozero.pickle")

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
cost2 = PathQsRCost(Q, R, x_path=x_path,u_path=u_path)
#
# # Random initial action path.
us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
#
J_hist = []
ilqr = iLQR(dynamics, cost2, N-1)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)

# max_bounds = 15.0
# min_bounds = -15.0
# diff = (max_bounds - min_bounds) / 2.0
# mean = (max_bounds + min_bounds) / 2.0
# us[:,0] = diff * np.tanh(us[:,0]) + mean
#

# diff = (max_bounds - min_bounds) / 2.0
# mean = (max_bounds + min_bounds) / 2.0
# us = diff * np.tanh(us) + mean

#
#
# max_bounds = 1.0
# min_bounds = -1.0
# diff = (max_bounds - min_bounds) / 2.0
# mean = (max_bounds + min_bounds) / 2.0
# us[:,2] = diff * np.tanh(us[:,2]) + mean
# us[:,5] = diff * np.tanh(us[:,2]) + mean

# #Constrain the actions to see what's actually applied to the system.
# controller = RecedingHorizonController(x0,ilqr)
# count =0
# plt.ion()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlim([0, 200])
# ax.set_ylim([-.8, -0.1])
# x = []
# y1 = []
# y2 = []
#
#
# line1, = ax.plot(x, y1, 'r-')
# line2, = ax.plot(x, y2, 'r-')
# for xs2, us2 in controller.control(us_init):
#     print(xs2)
#     y1.append(xs2[0][0])
#     y2.append(xs2[1][0])
#     x.append(count)
#     count+=1
#     line1.set_ydata(y1)
#     line2.set_ydata(y2)
#     line1.set_xdata(x)
#     line2.set_xdata(x)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
with open('test.npy', 'wb') as f:
    np.save(f, us)
print(len(us))
file = "/home/nathanielgoldfarb/linearize_model/test.npy"
with open(file, 'rb') as f:
    us2 = np.load(f)
y = x0
path = []
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