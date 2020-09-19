from ilqr.dynamics import FiniteDiffDynamics
import numpy as np
from ilqr import iLQR
from ilqr.cost import QRCost
import matplotlib.pyplot as plt



state_size = 2  # [position, velocity]
action_size = 1  # [force]

dt = 0.01  # Discrete time-step in seconds.
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

def f(x, u, i):

    [x, x_dot] = x
    [F] = u

    # Acceleration.
    x_dot_dot = x_dot * (1 - alpha * dt / m) + F * dt / m
    print
    return np.array([
        x + x_dot * dt,
        x_dot + x_dot_dot * dt,
    ])






dynamics = FiniteDiffDynamics(f, state_size, action_size)

state_size = 2  # [position, velocity]
action_size = 1  # [force]


Q = 100 * np.eye(state_size)
R = 0.01 * np.eye(action_size)

# This is optional if you want your cost to be computed differently at a
# terminal state.
Q_terminal = np.array([[100.0, 0.0], [0.0, 0.1]])

# State goal is set to a position of 1 m with no velocity.
x_goal = np.array([1.0, 0.0])

# NOTE: This is instantaneous and completely accurate.
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 1000  # Number of time-steps in trajectory.
x0 = np.array([0.0, -0.1])  # Initial state.
us_init = np.random.uniform(-1, 1, (N, 1)) # Random initial action path.

ilqr = iLQR(dynamics, cost, N)
xs, us = ilqr.fit(x0, us_init)
y = x0
path = []
for i in range(99):
    y = f(y, u_path[i], i)
    path.append(y)
plt.plot(xs[:,0])

plt.show()