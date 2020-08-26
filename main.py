import model
import numpy as np
import rbdl
import matplotlib.pyplot as plt


my_model = model.dynamic_model()
dt = 0.01
tf = 3
hip = model.get_traj(0.0, -0.3, 0.0, 0.0, tf, dt)
knee = model.get_traj(0.0, 0.20, 0.0, 0., tf, dt)
ankle = model.get_traj(-0.349, -0.1, 0.0, 0.0, tf, dt)

u = np.asarray([.1] * 6)

def nonlin():
    hip_traj = []
    knee_traj = []
    ankle_traj = []
    q = np.zeros(my_model.qdot_size)
    qd = np.zeros(my_model.qdot_size)
    y = np.concatenate((q, qd))

    for t in range(int(tf/dt)):
        print(t)
        q_d = np.array([hip["q"][t].item(), knee["q"][t].item(), ankle["q"][t].item(),
                          hip["q"][t].item(), knee["q"][t].item(), ankle["q"][t].item()])

        qd_d = np.array([hip["qd"][t].item(), knee["qd"][t].item(), ankle["qd"][t].item(),
                          hip["qd"][t].item(), knee["qd"][t].item(), ankle["qd"][t].item()])

        qdd_d = np.array([hip["qdd"][t].item(), knee["qdd"][t].item(), ankle["qdd"][t].item(),
                          hip["qdd"][t].item(), knee["qdd"][t].item(), ankle["qdd"][t].item()])

        tau = np.asarray([0.0] * 6)

        rbdl.InverseDynamics(my_model, q_d, qd_d, qdd_d, tau)

        y = model.runge_integrator(my_model, y, 0.01, tau)
        hip_traj.append(y[0])
        knee_traj.append(y[1])
        ankle_traj.append(y[2])
    return hip_traj, knee_traj, ankle_traj



def lin():
    hip_traj = []
    knee_traj = []
    ankle_traj = []
    q = np.zeros(my_model.qdot_size)
    qd = np.zeros(my_model.qdot_size)
    y = np.concatenate((q, qd))

    for t in range(int(tf/dt)):
        print(t)
        q_d = np.array([hip["q"][t].item(), knee["q"][t].item(), ankle["q"][t].item(),
                          hip["q"][t].item(), knee["q"][t].item(), ankle["q"][t].item()])

        qd_d = np.array([hip["qd"][t].item(), knee["qd"][t].item(), ankle["qd"][t].item(),
                          hip["qd"][t].item(), knee["qd"][t].item(), ankle["qd"][t].item()])

        qdd_d = np.array([hip["qdd"][t].item(), knee["qdd"][t].item(), ankle["qdd"][t].item(),
                          hip["qdd"][t].item(), knee["qdd"][t].item(), ankle["qdd"][t].item()])

        #tau = qdd_d # np.asarray([0.01] * 6)
        tau = np.asarray([0.0] * 6)
        traj = np.concatenate((q_d, qd_d))
        rbdl.InverseDynamics(my_model, q_d, qd_d, qdd_d, tau)

        A, b = model.finite_differences(my_model, traj, tau, h=0.01)
        y = model.runge_integrator_lin(A, b, y, 0.01, u)
        hip_traj.append(y[0] + q_d[0] )
        knee_traj.append(y[1] + q_d[1])
        ankle_traj.append(y[2] + q_d[2])
    return hip_traj, knee_traj, ankle_traj




hip_traj, knee_traj, ankle_traj = nonlin()
hip_traj_lin, knee_traj_lin, ankle_traj_lin = lin()
plt.plot(hip["q"])
plt.plot(hip_traj)
plt.plot(hip_traj_lin)
plt.legend(["traj", "nonlin", "lin"])
plt.show()