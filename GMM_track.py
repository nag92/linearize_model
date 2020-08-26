#!/usr/bin/env python
import rbdl
import model
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

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



def f(x, u, i):

    y = model.runge_integrator(my_model, x, 0.01, u)

    return np.array(y)

dynamics = FiniteDiffDynamics(f, 12, 6)



curr_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
curr_u = np.array([5.0, 2.0, 1.0,5.0, 2.0, 1.0])

runner = TPGMMRunner.TPGMMRunner("/home/nathaniel/catkin_ws/src/ambf_walker/config/gotozero.pickle")
x_path = []
u_path = []
N = int(tf/dt)
count = 0
while count < runner.get_length():
    count+=1
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
R = 0.00005 * np.eye(dynamics.action_size)
#
cost2 = PathQsRCost(Q, R, x_path=x_path,u_path=u_path)
#
# # Random initial action path.
us_init = np.random.uniform(-1, 1, (99, dynamics.action_size))
#
J_hist = []
ilqr = iLQR(dynamics, cost2, 99)
# xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)

controller = RecedingHorizonController(x0,ilqr)
count =0
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 200])
ax.set_ylim([-.8, -0.1])
x = []
y = []


line1, = ax.plot(x, y, 'r-')
for xs2, us2 in controller.control(us_init):
    print(xs2[0][0])
    y.append(xs2[0][0])
    x.append(count)
    count+=1
    line1.set_ydata(y)
    line1.set_xdata(x)
    fig.canvas.draw()
    fig.canvas.flush_events()


_ = plt.title("Trajectory of the two omnidirectional vehicles")
_ = plt.plot(x_path[:,0], "r")
_ = plt.plot(xs[:, 0], "b")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

plt.show()