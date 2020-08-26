#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics


# In[6]:


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


# In[7]:


dt = 0.02
pendulum_length = 1.0
dynamics = InvertedPendulumDynamics(dt, l=pendulum_length)


# In[9]:


# Note that the augmented state is not all 0.
x_goal = dynamics.augment_state(np.array([0.0, 0.0]))
Q = np.eye(dynamics.state_size)
Q[0, 1] = Q[1, 0] = pendulum_length
Q[0, 0] = Q[1, 1] = pendulum_length**2
Q[2, 2] = 0.0
Q_terminal = 100 * np.eye(dynamics.state_size)
R = np.array([[0.1]])
T = 100
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)


# In[ ]:


N = 300
x0 = dynamics.augment_state(np.array([np.pi, 0.0]))
us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))
ilqr = iLQR(dynamics, cost, N)


J_hist = []
xs, us = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


# In[ ]:


# Reduce the state to something more reasonable.
#xs = dynamics.reduce_state(xs)

# Constrain the actions to see what's actually applied to the system.
#us = constrain(us, dynamics.min_bounds, dynamics.max_bounds)


# In[ ]:


t = np.arange(N) * dt
theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
theta_dot = xs[:, 1]


# In[ ]:


_ = plt.plot(theta, theta_dot)
_ = plt.xlabel("theta (rad)")
_ = plt.ylabel("theta_dot (rad/s)")
_ = plt.title("Phase Plot")


# In[ ]:


_ = plt.plot(t, us)
_ = plt.xlabel("time (s)")
_ = plt.ylabel("Force (N)")
_ = plt.title("Action path")


# In[ ]:


_ = plt.plot(J_hist)
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Total cost")
_ = plt.title("Total cost-to-go")


# In[ ]:


print(xs)


# In[ ]:


xs[:,1]


# In[ ]:


plt.plot(xs[:,1])


# In[ ]:


plt.plot(xs[:,2])


# In[ ]:


plt.plot(xs[:,0])


# In[ ]:

plt.show()

