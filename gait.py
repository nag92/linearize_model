#!/usr/bin/env python
import rbdl
import model
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost, PathQsRCost

from ilqr.dynamics import AutoDiffDynamics, BatchAutoDiffDynamics, FiniteDiffDynamics


dt = 0.01  # Discrete time-step in seconds.
tf = 2.0
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.
from ilqr.dynamics import FiniteDiffDynamics
my_model = model.dynamic_model()
J_hist = []

def get_traj(q0, qf, v0, vf, tf, dt):

    b = np.array([q0, v0, qf, vf]).reshape((-1,1))
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [1.0, tf, tf ** 2, tf ** 3],
                  [0.0, 1.0, 2 * tf, 3 * tf * 2]])

    x = np.linalg.solve(A, b)
    q = []
    qd = []
    qdd = []

    for t in np.linspace(0, tf, int(tf/dt)):
        q.append(x[0] + x[1] * t + x[2] * t * t + x[3] * t * t * t)
        qd.append(x[1] + 2*x[2] * t + 3*x[3] * t * t)
        qdd.append(2*x[2] + 6*x[3] * t)

    traj = {}
    traj["q"] = q
    traj["qd"] = qd
    traj["qdd"] = qdd
    return traj


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)



def f(x, u, i):
    """Dynamics model function.

    Args:
        x: State vector [state_size].
        u: Control vector [action_size].
        i: Current time step.

    Returns:
        Next state vector [state_size].
    """


    y = model.runge_integrator(my_model, x, 0.01, u)

    return np.array(y)



# x_inputs=[
#     T.dscalar("Lhip"), #0
#     T.dscalar("Lknee"), #1
#     T.dscalar("Lankle"), #2
#     T.dscalar("Rhip"), #3
#     T.dscalar("Rknee"), #4
#     T.dscalar("Rankle"), #5
#     T.dscalar("Lhip_dot"), #6
#     T.dscalar("Lknee_dot"), #7
#     T.dscalar("Lankle_dot"), #8
#     T.dscalar("Rhip_dot"), #9
#     T.dscalar("Rknee_dot"), #10
#     T.dscalar("Rankle_dot"), #11
# ]
#
# u_inputs = [
#     T.dscalar("Lhip_ddot"),
#     T.dscalar("Lknee_ddot"),
#     T.dscalar("Lankle_ddot"),
#     T.dscalar("Rhip_ddot"),
#     T.dscalar("Rknee_ddot"),
#     T.dscalar("Rankle_ddot"),
# ]
#

dynamics = FiniteDiffDynamics(f, 12, 6)



curr_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
curr_u = np.array([5.0, 2.0, 1.0,5.0, 2.0, 1.0])
print(dynamics.f(curr_x, curr_u, 0))
x0 = [0.0, 0.0, -0.349,0.0, 0.0, -0.349, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
hip = model.get_traj(0.0, -0.3, 0.0, 0.0, tf, dt)
knee = model.get_traj(0.0, 0.20, 0.0, 0., tf, dt)
ankle = model.get_traj(-0.349, -0.1, 0.0, 0.0, tf, dt)

x_path = []
u_path = []
N = int(tf/dt)

for i in range(N):
    x_path.append([hip["q"][i][0], knee["q"][i][0], ankle["q"][i][0], hip["q"][i][0], knee["q"][i][0], ankle["q"][i][0],
                hip["qd"][i][0], knee["qd"][i][0], ankle["qd"][i][0],hip["qd"][i][0], knee["qd"][i][0], ankle["qd"][i][0] ])

for i in range(N-1):
    u_path.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


x_path = np.array(x_path)
u_path = np.array(u_path)
Q = np.eye(dynamics.state_size)*125.0
Q[6, 6] = Q[7, 7] = Q[8, 8] = Q[9, 9] = Q[10, 10] = Q[11, 11] = 0.0
Q = [Q]*N
R = 0.01 * np.eye(dynamics.action_size)

cost2 = PathQsRCost(Q, R, x_path=x_path,u_path=u_path)

# Random initial action path.
us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))

J_hist = []
ilqr = iLQR(dynamics, cost2, N-1)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)

_ = plt.title("Trajectory of the two omnidirectional vehicles")
_ = plt.plot(knee["q"], "r")
_ = plt.plot(xs[:,1], "b")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

plt.show()