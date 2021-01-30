
from . import model
import numpy as np
from ilqr import iLQR
from ilqr.cost import PathQsRCost
from ilqr.dynamics import FiniteDiffDynamics
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner



max_bounds = 10
max_bounds = -10

def f(x, us, i):

    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    u = diff * np.tanh(us) + mean
    y = model.runge_integrator(x, 0.01, u)

    return np.array(y)


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)

if __name__ == '__main__':

    model_path = ""
    model.make_dynamic_model("exo", model_path)

    dynamics = FiniteDiffDynamics(f, 12, 6)

    file_name = "/home/jack/backup/leg1.pickle"
    # file_name = "/home/jack/catkin_ws/src/ambf_walker/Train/gotozero.pickle"
    runner = TPGMMRunner.TPGMMRunner(file_name)

    x_path = []
    u_path = []

    # us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
    count = 0
    N = runner.get_length()

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
    Q = [np.zeros((size * 2, size * 2))] * len(expSigma)
    for ii in range(len(expSigma) - 2, -1, -1):
        Q[ii][:size, :size] = np.linalg.pinv(expSigma[ii])

    x0 = x_path[0]
    x_path = np.array(x_path)
    u_path = np.array(u_path)

    # R = 0.05 * np.eye(dynamics.action_size) # +/-10
    R = 5.0e-4 * np.eye(dynamics.action_size)

    R[3, 3] = 3.0e-3
    R[4, 4] = 3.0e-3
    R[5, 5] = 3.0e-3
    # R[1,1] = 0.00005
    # R[4,4] = 5.0e-6
    #
    # R[1,1] = 40.0
    # R[4,4] = 40.0

    # print(x_path)

    print(R)

    # print(u_path)

    us_init = np.random.uniform(-1, 1, (N - 1, dynamics.action_size))
    #
    #
    cost = PathQsRCost(Q, R, x_path=x_path, u_path=u_path)

    # print(cost)
    #
    # # Random initial action path.

    #
    J_hist = []
    ilqr = iLQR(dynamics, cost, N - 1)
    xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)
