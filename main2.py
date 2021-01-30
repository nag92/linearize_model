import model
import numpy as np
import rbdl
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner

my_model = model.dynamic_model()



Kp = np.zeros((6, 6))
Kd = np.zeros((6, 6))

Kp_hip = 100.0
Kd_hip = 20.0

Kp_knee = 2.0
Kd_knee = 0.0

Kp_ankle = 100.0
Kd_ankle = 0.0

Kp[0, 0] = Kp_hip
Kd[0, 0] = Kd_hip

Kp[1, 1] = Kp_knee
Kd[1, 1] = Kd_knee

Kp[2, 2] = Kp_ankle
Kd[2, 2] = Kd_ankle

Kp[3, 3] = Kp_hip
Kd[3, 3] = Kd_hip

Kp[4, 4] = Kp_knee
Kd[4, 4] = Kd_knee

Kp[5, 5] = Kp_ankle
Kd[5, 5] = Kd_ankle

def nonlin():
    hip_traj = []
    path = []
    runner = TPGMMRunner.TPGMMRunner("/home/nathaniel/catkin_ws/src/ambf_walker/config/gotozero.pickle")
    count = 0
    x = runner.get_start()
    v0 = np.zeros(len(x)).reshape((-1, 1))
    y = np.concatenate((x, v0))

    while count < runner.get_length():
        runner.step()
        u = runner.ddx
        x = runner.x
        dx = runner.dx
        u = Kp.dot(x - y[0:6]) + Kd.dot(dx - y[6:])
        tau = np.asarray([0.0] * 6)
        rbdl.InverseDynamics(my_model, x.flatten(), dx.flatten(), u.flatten(), tau)
        y = model.runge_integrator(my_model, y.flatten(), 0.01, tau.flatten())
        hip_traj.append(y[0])

        count += 1
        path.append(x[0])
    return hip_traj, path



def lin():
    hip_traj = []
    path = []
    runner = TPGMMRunner.TPGMMRunner("/home/nathaniel/catkin_ws/src/ambf_walker/config/gotozero.pickle")
    count = 0
    x = runner.get_start()
    v0 = np.zeros(len(x)).reshape((-1, 1))
    y = np.concatenate((x, v0))

    while count < runner.get_length():
        runner.step()
        u = runner.ddx
        x = runner.x
        dx = runner.dx
        # u = Kp.dot(x - y[0:6]) + Kd.dot(dx - y[6:])
        tau = np.asarray([0.0] * 6)
        rbdl.InverseDynamics(my_model, x.flatten(), dx.flatten(), u.flatten(), u.flatten())
        du = np.asarray([.1] * 6)
        traj = np.concatenate((x, dx))
        A, b = model.finite_differences(my_model, traj.flatten(), tau, h=0.01)
        y = model.runge_integrator_lin(A, b, y.flatten(), 0.01, du)
        hip_traj.append(y[0]+x[0])

        count += 1
        path.append(x[0])

    return hip_traj, path



hip_traj, path = nonlin()
hip_traj_lin, _ = lin()
#plt.plot(path)
#plt.plot(hip_traj)
plt.plot(hip_traj_lin)
plt.legend(["path", "nonlin"])
plt.show()