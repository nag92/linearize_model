#!/usr/bin/env python
import model
import numpy as np
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from ilqr import iLQR
from ilqr.controller import RecedingHorizonControllerPath
from ilqr.cost import PathQsRCost, PathQRCostMPC
from ilqr.dynamics import FiniteDiffDynamics
from tqdm import tqdm
from scipy.stats import entropy
import gc
from itertools import product
from multiprocessing import Pool
import scipy.io

dt = 0.01  # Discrete time-step in seconds.
tf = 2.0
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

my_model = model.dynamic_model()
J_hist = []


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)

# max_bounds = 10.0
# min_bounds = -10.0


##### 8 and -8
# max_bounds = 10.0
# min_bounds = -10.0

def measure_d(list_1, list_2):
    #### list length should be same
    #### change to RMS


    N = len(list_1)
    D = 0

    for i_list in range(N):
        d = (list_1[i_list] - list_2[i_list])**2
        D = D + d

    D = np.sqrt(D/N)

    return D


def f(x, us, i):

    # diff = (max_bounds - min_bounds) / 2.0
    # mean = (max_bounds + min_bounds) / 2.0
    # us = diff * np.tanh(u) + mean
    y = model.runge_integrator(my_model, x, 0.01, us)

    return np.array(y)

def f2(x, u, i):

    y = model.runge_integrator(my_model, x, 0.01, u)

    return np.array(y)
############################################## MODIFY ####################

###### original : return np.array(y)



dynamics = FiniteDiffDynamics(f, 12, 6)

file_name = "./leg1.pickle"
#file_name = "/home/jack/catkin_ws/src/ambf_walker/Train/gotozero.pickle"
runner = TPGMMRunner.TPGMMRunner(file_name)

x_path = []
u_path = []

#us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
count = 0
N = runner.get_length()
# print(N)
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
Q = [np.zeros((size*2, size*2))]*len(expSigma)
for ii in range(len(expSigma)-2, -1, -1):
    Q[ii][:size, :size] = np.linalg.pinv(expSigma[ii])

x0 = x_path[0]
x_path = np.array(x_path)
u_path = np.array(u_path)

# R = 0.05 * np.eye(dynamics.action_size) # +/-10
R = np.eye(dynamics.action_size)

# R[0,0] = 3.0e-3
# R[1,1] = 3.0e-3
# R[2,2] = 3.0e-3
# R[3,3] = 3.0e-3
# R[4,4] = 3.0e-3
# R[5,5] = 3.0e-3


# print(x_path)

# print(R)

# KL0 = 1.0e5

def main_fuc(R0):
    # print(R0)
    R1 = R0[0]
    R2 = R0[1]
    R3 = R0[2]
    R[0,0] = R1
    R[1,1] = R2
    R[2,2] = R3
    R[3,3] = R1
    R[4,4] = R2
    R[5,5] = R3
    us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
    cost = PathQsRCost(Q, R, x_path=x_path, u_path=u_path)
    #print(cost)
    J_hist = []
    ilqr = iLQR(dynamics, cost, N-1)
    xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)
    # len_us = len(us)
    # print(f"len(us)={len_us}")
    # print(us)
    # print(xs)
    # print(us)
    cost2 = PathQsRCost(Q, R, x_path, us)
    ilqr2 = iLQR(dynamics, cost2, N-1)
    cntrl = RecedingHorizonControllerPath(x0, ilqr2)
    # print(J_hist)
    ####### if want to smooth J_hist
    # plt.figure()
    # plt.plot(J_hist)
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.grid()
    # plt.show(

    plt.ion()
    count = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax1.set_xlim([0, 177])
    ax1.set_ylim([-0.4, 0.6])
    ax2 = fig.add_subplot(323)
    ax2.set_xlim([0, 177])
    ax2.set_ylim([-0.2, 2.0])
    ax3 = fig.add_subplot(325)
    ax3.set_xlim([0, 177])
    ax3.set_ylim([-0.3, 0.3])
    ax4 = fig.add_subplot(322)
    ax4.set_xlim([0, 177])
    ax4.set_ylim([-0.8, 0.6])
    ax5 = fig.add_subplot(324)
    ax5.set_xlim([0, 177])
    ax5.set_ylim([0.0, 1.7])
    ax6 = fig.add_subplot(326)
    ax6.set_xlim([0, 177])
    ax6.set_ylim([-0.3, 0.3])
    x1 = []
    y1 = []
    x1_follow = []
    y1_follow = []
    x2 = []
    y2 = []
    x2_follow = []
    y2_follow = []
    x3 = []
    y3 = []
    x3_follow = []
    y3_follow = []
    x4 = []
    y4 = []
    x4_follow = []
    y4_follow = []
    x5 = []
    y5 = []
    x5_follow = []
    y5_follow = []
    x6 = []
    y6 = []
    x6_follow = []
    y6_follow = []
    us3 = []
    line1, = ax1.plot(x1, y1, 'r-')
    line2, = ax1.plot(x1_follow, y1_follow, 'b-')
    line3, = ax2.plot(x2, y2, 'r-')
    line4, = ax2.plot(x2_follow, y2_follow, 'b-')
    line5, = ax3.plot(x3, y3, 'r-')
    line6, = ax3.plot(x3_follow, y3_follow, 'b-')
    line7, = ax4.plot(x4, y4, 'r-')
    line8, = ax4.plot(x4_follow, y4_follow, 'b-')
    line9, = ax5.plot(x5, y5, 'r-')
    line10, = ax5.plot(x5_follow, y5_follow, 'b-')
    line11, = ax6.plot(x6, y6, 'r-')
    line12, = ax6.plot(x6_follow, y6_follow, 'b-')

    #print(sys.getsizeof(cntrl.control(us)))
    #
    # with open('test.npy', 'wb') as f:
    #     np.save(f, us)
    # print(len(us))
    # print('step1')

    for xs2, us2 in tqdm(cntrl.control(us)):
        # print(us2[0])
        # print(type(us2[0]))
        us3.append(us2[0])
        x1.append(count)
        y1.append(xs2[0][0])
        x1_follow.append(count)
        y1_follow.append(x_path[count][0])
        x2.append(count)
        y2.append(xs2[0][1])
        x2_follow.append(count)
        y2_follow.append(x_path[count][1])
        x3.append(count)
        y3.append(xs2[0][2])
        x3_follow.append(count)
        y3_follow.append(x_path[count][2])
        x4.append(count)
        y4.append(xs2[0][3])
        x4_follow.append(count)
        y4_follow.append(x_path[count][3])
        x5.append(count)
        y5.append(xs2[0][4])
        x5_follow.append(count)
        y5_follow.append(x_path[count][4])
        x6.append(count)
        y6.append(xs2[0][5])
        x6_follow.append(count)
        y6_follow.append(x_path[count][5])
        # print(xs2[0][0:6])
        # print(x_path[count][0:6])
        count += 1
        cntrl.set_state(xs2[1])
        line1.set_ydata(y1)
        line1.set_xdata(x1)
        line2.set_ydata(y1_follow)
        line2.set_xdata(x1_follow)
        line3.set_ydata(y2)
        line3.set_xdata(x2)
        line4.set_ydata(y2_follow)
        line4.set_xdata(x2_follow)
        line5.set_ydata(y3)
        line5.set_xdata(x3)
        line6.set_ydata(y3_follow)
        line6.set_xdata(x3_follow)
        line7.set_ydata(y4)
        line7.set_xdata(x4)
        line8.set_ydata(y4_follow)
        line8.set_xdata(x4_follow)
        line9.set_ydata(y5)
        line9.set_xdata(x5)
        line10.set_ydata(y5_follow)
        line10.set_xdata(x5_follow)
        line11.set_ydata(y6)
        line11.set_xdata(x6)
        line12.set_ydata(y6_follow)
        line12.set_xdata(x6_follow)
        fig.canvas.draw()
        fig.canvas.flush_events()
        us = us[1:]
        if count == 177:
            break
    plt.ioff()
    # plt.show()

    KL1 = measure_d(y1_follow, y1)
    KL2 = measure_d(y2_follow, y2)
    KL3 = measure_d(y3_follow, y3)
    KL4 = measure_d(y4_follow, y4)
    KL5 = measure_d(y5_follow, y5)
    KL6 = measure_d(y6_follow, y6)

    KL = KL1 + KL2 + KL3 + KL4 + KL5 + KL6

    fname = f'R1_{R1}_R2_{R2}_R3_{R3}_KL_{KL}.png'
    s_path = f'./data2/{fname}'
    dname = f'R1_{R1}_R2_{R2}_R3_{R3}_KL_{KL}.mat'
    d_path = f'./data2/{dname}'
    plt.savefig(s_path)
    scipy.io.savemat(d_path,mdict={'y1':y1, 'y2':y2, 'y3':y3, 'y4':y4, 'y5':y5, 'y6':y6,'y1_follow':y1_follow, 'y2_follow':y2_follow, 'y3_follow':y3_follow, 'y4_follow':y4_follow, 'y5_follow':y5_follow, 'y6_follow':y6_follow, 'torque':us3})

    plt.close('all')

    print(KL)

    # if KL > KL0:
    #     KL = KL0

    gc.collect()
    return KL

# for R1 in R_range:
#     # R1 = R1 * 1.0e-4
#     for R2 in R_range:
#         # R2 = R2 * 1.0e-4
#         KL0 = main_fuc(R1,R2, KL0)
#         # KL = main_fuc(R_1, R_2)
#         # if KL > KL0:
#         #     KL = KL0

if __name__ == '__main__':
    dt = 0.01  # Discrete time-step in seconds.
    tf = 2.0
    m = 1.0  # Mass in kg.
    alpha = 0.1  # Friction coefficient.
    my_model = model.dynamic_model()
    J_hist = []
    dynamics = FiniteDiffDynamics(f, 12, 6)
    file_name = "./leg1.pickle"  ##### mayreplace this file
    runner = TPGMMRunner.TPGMMRunner(file_name)
    x_path = []
    u_path = []
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
    Q = [np.zeros((size*2, size*2))]*len(expSigma)
    for ii in range(len(expSigma)-2, -1, -1):
        Q[ii][:size, :size] = np.linalg.pinv(expSigma[ii])
    x0 = x_path[0]
    x_path = np.array(x_path)
    u_path = np.array(u_path)
    # R = 0.05 * np.eye(dynamics.action_size) # +/-10
    R = np.eye(dynamics.action_size)

    R_range1 = [1e-4,2e-4,3e-4,4e-4,5e-6,6e-6,7e-6,8e-6,9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5]
    R_range2 = [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-5,6e-5,7e-5,8e-5,9e-5]
    R_range3 = [1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1,2e-1,3e-1,4e-1,5e-3,6e-3,7e-3,8e-3,9e-3]
    # R_range = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2]
    # R_m = product(R_range, repeat = 3)

    R_m = []

    for R1_ele in R_range1:
        for R2_ele in R_range2:
            for R3_ele in R_range3:
                R_m.append((R1_ele,R2_ele,R3_ele))

    R_m = iter(R_m)

    with Pool(16) as p:
        ALL_KL=p.map(main_fuc,R_m)

print(ALL_KL)
scipy.io.savemat('all_kl_2.mat',mdict={'All_KL':ALL_KL})
