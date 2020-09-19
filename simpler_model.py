#!/usr/bin/env python
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

from ilqr import iLQR, RecedingHorizonController
from ilqr.cost import QRCost, PathQRCost, Cost
from ilqr.dynamics import AutoDiffDynamics


def get_traj(q0, qf, v0, vf, tf, dt):
    b = np.array([q0, v0, qf, vf]).reshape((-1, 1))
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [1.0, tf, tf ** 2, tf ** 3],
                  [0.0, 1.0, 2 * tf, 3 * tf ** 2]])

    x = np.linalg.solve(A, b)
    q = []
    qd = []
    qdd = []

    for t in np.linspace(0, tf, int(tf / dt)):
        q.append(x[0] + x[1] * t + x[2] * t * t + x[3] * t * t * t)
        qd.append(x[1] + 2 * x[2] * t + 3 * x[3] * t * t)
        qdd.append(2 * x[2] + 6 * x[3] * t)

    traj = {}
    traj["q"] = q
    traj["qd"] = qd
    traj["qdd"] = qdd

    return traj


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)



x_path = get_traj(10.0, 3.0, 5.0, 0, 2.0, 0.01)
y_path = get_traj(10.0, 5.0, 0, 0, 2.0, 0.01)

traj = []
for i in range(len(x_path["q"])):
    traj.append( np.array( [x_path["q"][i][0], y_path["q"][i][0],x_path["qd"][i][0], y_path["qd"][i][0] ] ) )

traj = np.array(traj)

x_inputs = [
    T.dscalar("x_0"),
    T.dscalar("y_0"),
    T.dscalar("x_0_dot"),
    T.dscalar("y_0_dot"),

]

u_inputs = [
    T.dscalar("F_x_0"),
    T.dscalar("F_y_0"),
]

dt = 0.1  # Discrete time step.
m = 1.0  # Mass.
alpha = 0.1  # Friction coefficient.


# Acceleration.
def acceleration(x_dot, u):
    x_dot_dot = x_dot * (1 - alpha * dt / m) + u * dt / m
    return x_dot_dot


# Discrete dynamics model definition.
f = T.stack([
    x_inputs[0] + x_inputs[2] * dt,
    x_inputs[1] + x_inputs[3] * dt,
    x_inputs[2] + acceleration(x_inputs[2], u_inputs[0]) * dt,
    x_inputs[3] + acceleration(x_inputs[3], u_inputs[1]) * dt,
])

dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)

Q = np.eye(dynamics.state_size)
# Q[0, 2] = Q[2, 0] = -1
# Q[1, 3] = Q[3, 1] = -1
R = 0.1 * np.eye(dynamics.action_size)

N = 199  # Number of time steps in trajectory.
x0 = np.array([10.0, 10.0, 5.0, 0.0])  # Initial state.


# Random initial action path.
us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))


cost = PathQRCost(Q, R, traj, us_init)

J_hist = []
ilqr = iLQR(dynamics, cost, N)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)

cost2 = PathQRCost(Q, R, xs, us)

ilqr2 = iLQR(dynamics, cost2, N)

cntrl = RecedingHorizonController(x0, ilqr2)
plt.ion()

count = 0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-3, 30])
ax.set_ylim([-3, 30])
x = []
y = []

line1, = ax.plot(x, y, 'r-')
for xs2, us2 in cntrl.control(us_init):
    x.append(xs2[0][0])
    y.append(xs2[0][1])
    cntrl.set_state(xs2[1])
    count += 1
    line1.set_ydata(y)
    line1.set_xdata(x)
    fig.canvas.draw()
    fig.canvas.flush_events()
#
# x_0 = xs[:, 0]
# y_0 = xs[:, 1]
# x_0_dot = xs[:, 2]
# y_0_dot = xs[:, 3]

# In[11]:

x_0 = xs[:, 0]
y_0 = xs[:, 1]
x_0_dot = xs[:, 2]
y_0_dot = xs[:, 3]
_ = plt.title("Trajectory of the two omnidirectional vehicles")
_ = plt.plot(x_0, y_0, "r")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

plt.show()
# In[12]:


t = np.arange(N + 1) * dt
_ = plt.plot(t, x_0, "r")
_ = plt.xlabel("Time (s)")
_ = plt.ylabel("x (m)")
_ = plt.title("X positional paths")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

# In[13]:


_ = plt.plot(t, y_0, "r")
_ = plt.xlabel("Time (s)")
_ = plt.ylabel("y (m)")
_ = plt.title("Y positional paths")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

# In[14]:


_ = plt.plot(t, x_0_dot, "r")
_ = plt.xlabel("Time (s)")
_ = plt.ylabel("x_dot (m)")
_ = plt.title("X velocity paths")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

# In[15]:


_ = plt.plot(t, y_0_dot, "r")
_ = plt.xlabel("Time (s)")
_ = plt.ylabel("y_dot (m)")
_ = plt.title("Y velocity paths")
_ = plt.legend(["Vehicle 1", "Vehicle 2"])

# In[16]:


_ = plt.plot(J_hist)
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Total cost")
_ = plt.title("Total cost-to-go")
