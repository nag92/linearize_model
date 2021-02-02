
import numpy as np
import rbdl
import rospy
from rbdl_server.srv import RBDLModel, RBDLModelAlignment
from rbdl_server.srv import RBDLForwardDynamics

names = ['ExoLeftHip', 'ExoLeftKnee', 'ExoLeftAnkle',
         'ExoRightHip', 'ExoRightKnee', 'ExoRightAnkle']



def get_map():

    try:
        model_srv = rospy.ServiceProxy('AMBF2RBDL', RBDLModelAlignment)
        resp1 = model_srv("exo", [])
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

    return {resp1.names[i]: resp1.ids[i] for i in range(len(resp1.names))}


def ambf_to_rbdl(q,join_map):
    """
    make the order of the joints for the dynamics
    """

    joints_aligned = [0.0] * len(names)

    for ii, name in enumerate(names):
        index = joint_map[name] - 1
        joints_aligned[index] = q[ii]

    return joints_aligned


def rbdl_to_ambf(q,joint_map):
    """
    reverse the order of the AMBF
    """

    q_new = [0.0] * len(names)

    for ii, name in enumerate(names):
        index = joint_map[name] - 1

        q_new[ii] = q[index]

    return q_new


def make_dynamic_model(name, model_path):
    """"
    use the RBDL server to create the model
    """
    try:
        model_srv = rospy.ServiceProxy('CreateModel', RBDLModel)
        resp1 = model_srv(name, model_path)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def rhs(y, tau):

    dim = 6
    res = np.zeros(dim * 2)
    Q = np.zeros(dim)
    QDot = np.zeros(dim)
    QDDot = np.zeros(dim)
    Tau = tau
    for i in range(0, dim):
        Q[i] = y[i]
        QDot[i] = y[i + dim]

    rospy.wait_for_service("ForwardDynamics")
    try:
        dyn_srv = rospy.ServiceProxy('ForwardDynamics', RBDLForwardDynamics)
        resp1 = dyn_srv("exo", Q, QDot, Tau)
        QDDot = np.array(resp1.qdd)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

    # rbdl.ForwardDynamics(model, Q, QDot, Tau, QDDot)

    for i in range(0, dim):
        res[i] = QDot[i]
        res[i + dim] = QDDot[i]

    return res

def runge_integrator(y, h, tau):

    k1 = rhs(y, tau)
    k2 = rhs(y + 0.5 * h * k1, tau)
    k3 = rhs(y + 0.5 * h * k2, tau)
    k4 = rhs(y + h * k3, tau)

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