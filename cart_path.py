#!/usr/bin/env python





# In[3]:


import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt


# In[4]:


from ilqr import iLQR, RecedingHorizonController
from ilqr.cost import QRCost, PathQRCost, PathQsRCost
from ilqr.dynamics import AutoDiffDynamics


# In[5]:
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


# In[6]:


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





# The vehicles are initialized at $(0, 0)$ and $(10, 10)$ with velocities $(0, -5)$ and $(5, 0)$ respectively.

# In[8]:


N = 200  # Number of time steps in trajectory.
x0 = np.array([10.0, 10.0, 5.0, 3.0 ])  # Initial state.

x_traj =get_traj(10.0, 5.0, 5.0, 0.0, 2.0, 0.01)
y_traj =get_traj(10.0, 5.0, 3.0, 0.0, 2.0, 0.01)
x_path = []
u_path = []
for i in range(N):
    x_path.append([x_traj["q"][i][0], y_traj["q"][i][0], x_traj["qd"][i][0], y_traj["qd"][i][0] ])

for i in range(N-1):
    #u_path.append([x_traj["qdd"][i][0], y_traj["qdd"][i][0]])
    u_path.append([0.0, 0.0])

x_path = np.array(x_path)
u_path = np.array(u_path)
Q = np.eye(dynamics.state_size)*125.0
Q[2,2] = Q[3,3] = 0
#Q = [Q]*N
R = 0.01 * np.eye(dynamics.action_size)

cost = QRCost(Q, R)
#cost = PathQsRCost(Q,R,x_path=x_path,u_path=u_path)

# Random initial action path.
us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))


# In[9]:


J_hist = []
ilqr = iLQR(dynamics, cost, N-1)
#xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)


controller = RecedingHorizonController(x0,ilqr)
count =0
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([4, 12])
ax.set_ylim([5, 12])
x = []
y = []


line1, = ax.plot(x, y, 'r-')
for xs2, us2 in controller.control(us_init):
    print(xs2[1][1])
    x.append(xs2[0][0])
    y.append(xs2[0][1])
    count+=1
    line1.set_ydata(y)
    line1.set_xdata(x)
    fig.canvas.draw()
    fig.canvas.flush_events()



x_0 = xs[:, 0]
y_0 = xs[:, 1]
x_0_dot = xs[:, 2]
y_0_dot = xs[:, 3]


# In[11]:


_ = plt.title("Trajectory of the two omnidirectional vehicles")
_ = plt.plot(x_0, y_0, "r")
_ = plt.plot(x_path[:,0], x_path[:,1], "b")
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
