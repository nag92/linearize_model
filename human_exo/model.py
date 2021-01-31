import rbdl
import numpy as np

def dynamic_model(noise=0.0):
    # add in mass and height params
    model = rbdl.Model()
    bodies = {}
    mass = {}
    com = {}
    inertia = {}
    bodies["right"] = {}
    bodies["left"] = {}
    segments = ["thigh", "shank", "foot"]

    mass["hip"] = 45.32+noise
    mass["right_thigh"] = 13.934+noise
    mass["left_thigh"] = 13.934+noise
    mass["right_shank"] = 5.128+noise
    mass["left_shank"] = 5.128+noise
    mass["right_foot"] = 1.898+noise
    mass["left_foot"] = 1.898+noise

    
    parent_dist = {}
    parent_dist["hip"] = np.array([0.0, 0.0, 0.0])

    parent_dist["left_thigh"] = np.array([0.237, -0.124, -0.144])
    parent_dist["left_shank"] = np.array([0.033, -0.03, -0.436])
    parent_dist["left_foot"] = np.array([0.02, -0.027, -0.39])

    parent_dist["right_thigh"] = np.array([-0.237, -0.124, -0.144])
    parent_dist["right_shank"] = np.array([0.033, -0.03, -0.436])
    parent_dist["right_foot"] = np.array([0.02, -0.027, -0.39])

    inertia["hip"] = np.diag([5.9172, 6.1912, 1.4393])

    inertia["left_thigh"] = np.diag([0.5435, 0.6347, 0.1249])
    inertia["left_shank"] = np.diag([0.1365, 0.1843, 0.0562])
    inertia["left_foot"] = np.diag([0.0312, 0.0276, 0.0405])

    inertia["right_thigh"] = np.diag([0.5435, 0.6347, 0.1249])
    inertia["right_shank"] = np.diag([0.1365, 0.1843, 0.0562])
    inertia["right_foot"] = np.diag([0.0312, 0.0276, 0.0405])


    com["hip"] = np.array([0.00, -0.0785, 0.2803])
    com["left_thigh"] = np.array([-0.0833, -0.0198, -0.00678])
    com["left_shank"] = np.array([-0.0968, -0.007, 0.06])
    com["left_foot"] = np.array([-0.0733, -0.0583, 0.04])

    com["right_thigh"] = np.array([-0.0833, -0.0198, -0.00678])
    com["right_shank"] = np.array([-0.0968, -0.007, 0.06])
    com["right_foot"] = np.array([-0.0733, -0.0583, 0.04])

    hip_body = rbdl.Body.fromMassComInertia(mass["hip"], com["hip"], inertia["hip"])
    for segs in segments:
        bodies["right_" + segs] = rbdl.Body.fromMassComInertia(mass["right_" + segs], com["right_" + segs],
                                                               inertia["right_" + segs])
        bodies["left_" + segs] = rbdl.Body.fromMassComInertia(mass["left_" + segs], com["left_" + segs],
                                                              inertia["left_" + segs])

    xtrans = rbdl.SpatialTransform()
    xtrans.r = np.array([0.0, 0.0, 0.0])
    xtrans.E = np.eye(3)

    hip = model.AddBody(0, xtrans, rbdl.Joint.fromJointType("JointTypeFixed"), hip_body, "hip")
    joint_rot_z = rbdl.Joint.fromJointType("JointTypeRevoluteX")

    xtrans.r = parent_dist["left_thigh"]
    left_thigh = model.AddBody(hip, xtrans, joint_rot_z, bodies["left_thigh"], "left_thigh")
    xtrans.E = np.eye(3)
    xtrans.r = parent_dist["left_shank"]
    left_shank = model.AddBody(left_thigh, xtrans, joint_rot_z, bodies["left_shank"], "left_shank")
    xtrans.r = parent_dist["left_foot"]
    left_foot = model.AddBody(left_shank, xtrans, joint_rot_z, bodies["left_foot"], "left_foot")

    xtrans.r = parent_dist["right_thigh"]
    right_thigh = model.AddBody(hip, xtrans, joint_rot_z, bodies["right_thigh"], "right_thigh")
    xtrans.E = np.eye(3)
    xtrans.r = parent_dist["right_shank"]
    right_shank = model.AddBody(right_thigh, xtrans, joint_rot_z, bodies["right_shank"], "right_shank")
    xtrans.r = parent_dist["right_foot"]
    right_foot = model.AddBody(right_shank, xtrans, joint_rot_z, bodies["right_foot"], "right_foot")

    model.gravity = np.array([0, 0, -9.81])


    return model

def rhs(model, y, tau):

    dim = model.dof_count
    res = np.zeros(dim * 2)
    Q = np.zeros(model.q_size)
    QDot = np.zeros(model.qdot_size)
    QDDot = np.zeros(model.qdot_size)
    Tau = tau
    for i in range(0, dim):
        Q[i] = y[i]
        QDot[i] = y[i + dim]

    rbdl.ForwardDynamics(model, Q, QDot, Tau, QDDot)

    for i in range(0, dim):
        res[i] = QDot[i]
        res[i + dim] = QDDot[i]

    return res

def runge_integrator(model, y, h, tau):

    k1 = rhs(model, y, tau)
    k2 = rhs(model, y + 0.5 * h * k1,tau)
    k3 = rhs(model, y + 0.5 * h * k2,tau)
    k4 = rhs(model, y + h * k3,tau)

    return y + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)




def rhs_lin(A,b, y, tau):

    dim = 6
    res = np.zeros(dim * 2)
    QDot = np.zeros(dim)
    Q = np.zeros(dim)
    for i in range(0, dim):
        Q[i] = y[i]
        QDot[i] = y[i + dim]

    QDDot = A.dot(y.reshape((-1,1))) + b.dot(tau.reshape((-1,1)))

    for i in range(0, dim):
        res[i] = QDot[i]
        res[i + dim] = QDDot[i]

    return res

def runge_integrator_lin(A, b, y, h, tau):

    k1 = rhs_lin(A, b, y, tau)
    k2 = rhs_lin(A, b, y + 0.5 * h * k1, tau)
    k3 = rhs_lin(A, b, y + 0.5 * h * k2, tau)
    k4 = rhs_lin(A, b, y + h * k3, tau)

    return y + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


def get_traj(q0, qf, v0, vf, tf, dt):

    b = np.array([q0, v0, qf, vf]).reshape((-1,1))
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [1.0, tf, tf ** 2, tf ** 3],
                  [0.0, 1.0, 2 * tf, 3 * tf ** 2]])

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




def finite_differences(model, y, u, h=0.01):
    """ calculate gradient of plant dynamics using finite differences
    x np.array: the state of the system
    u np.array: the control signal
    """

    dof = u.shape[0]
    num_states = model.q_size*2

    A = np.zeros((num_states, num_states))
    B = np.zeros((num_states, dof))

    eps = 1e-4 # finite differences epsilon
    for ii in range(num_states):
        # calculate partial differential w.r.t. x
        inc_x = y.copy()
        inc_x[ii] += eps
        state_inc = runge_integrator(model, inc_x, h, u)
        dec_x = y.copy()
        dec_x[ii] -= eps
        state_dec = runge_integrator(model, dec_x, h, u)
        A[:, ii] = (state_inc - state_dec) / (2 * eps)

    for ii in range(dof):
        # calculate partial differential w.r.t. u
        inc_u = u.copy()
        inc_u[ii] += eps
        state_inc = runge_integrator(model, y, h, inc_u)
        dec_u = u.copy()
        dec_u[ii] -= eps
        state_dec = runge_integrator(model, y, h, dec_u)
        B[:, ii] = (state_inc - state_dec) / (2 * eps)

    return A, B
