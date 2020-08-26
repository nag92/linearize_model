import theano.tensor as T
from ilqr.dynamics import AutoDiffDynamics
from ilqr.cost import PathQRCost
import numpy as np
from ilqr import iLQR
import matplotlib.pyplot as plt
J_hist = []
def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)


dt = 0.01
x = T.dscalar("x")
u = T.dscalar("u")
t = T.dscalar("t")

x_dot = (dt * t - u) * x**2
f = T.stack([x + x_dot * dt])

dynamics = AutoDiffDynamics(f, [x], [u], t)


T = 200
state_size = 1
action_size = 1

Q = np.eye(state_size)
R = np.eye(action_size)

dist = np.linspace(0, 3.14, T + 1).reshape(-1, 1)
x_path = np.sin(dist)
dist = np.linspace(0, 3.14, T ).reshape(-1, 1)
u_path = -np.cos(dist)

cost = PathQRCost(Q, R, x_path, u_path=u_path)

from ilqr.dynamics import tensor_constrain, constrain

# When building dynamics model...

# When fitting..
ilqr = iLQR(dynamics, cost, T)
us_init = np.random.uniform(-1, 1, (T, dynamics.action_size))
xs, us_unconstrained = ilqr.fit([0.0], us_init,n_iterations=10000,on_iteration=on_iteration)
print(xs)