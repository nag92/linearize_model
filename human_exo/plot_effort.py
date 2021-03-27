#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy.io
import scipy.interpolate
from glob import glob

def RMS_error(list_1, list_2):
    #### list length should be same
    #### change to RMS

    N = len(list_1)
    D = 0

    for i_list in range(N):
        d = (list_1[i_list] - list_2[i_list])**2
        D = D + d

    D = np.sqrt(D/N)

    return D

# def value_m(M_in):
#     M_out = np.zeros(len(M_in))
#     for i in range(len(M_in)):
#         if M_in[i] > 9.8e-4:
#             M_out[i] = M_in[i]/10 + 9e-4
#         else:
#             M_out[i] = M_in[i]
#     return M_out
#
#
# file_names = os.listdir('./data0')

file_names = glob('./data0/*.mat')

s_files = []
data_files = []
R1_values = []
R2_values = []
R3_values = []
KL_values = []
R_matrix = []
u_torque = []
data = []
Effort = []

for i_name in file_names:
    D = scipy.io.loadmat(i_name)
    data.append(D)

for s_list in file_names:
    s_list = s_list.split('.mat')
    s_files.append(s_list[0])


for i_list in s_files:
    i_list = i_list.split('_')
    data_files.append(i_list)
    R1_values.append(float(i_list[1]))
    R2_values.append(float(i_list[3]))
    R3_values.append(float(i_list[5]))
    KL_values.append(float(i_list[7]))

R1 = np.asarray(R1_values)
R2 = np.asarray(R2_values)
R3 = np.asarray(R3_values)
KL = np.asarray(KL_values)

min_KL = np.amin(KL)
min_KL_index = int(np.where(KL == np.amin(KL))[0])

for i_Rterm in range(len(R1_values)):
    R_matrix.append(np.diag([R1[i_Rterm],R2[i_Rterm],R3[i_Rterm],R1[i_Rterm],R2[i_Rterm],R3[i_Rterm]]))
    u_torque.append(data[i_Rterm]['torque'])

for i_iter in range(len(KL_values)):
    U = u_torque[i_iter]
    R = R_matrix[i_iter]
    # R[1,1] = 0
    # R[2,2] = 0
    # R[4,4] = 0
    # R[5,5] = 0
    # U[:,1] = 0
    # U[:,2] = 0
    # U[:,4] = 0
    # U[:,5] = 0
    S = np.dot(np.dot(U,R),np.transpose(U))/(177*177)
    Effort.append(np.sum(S))

plt.figure()

plt.plot(KL,Effort,'.')
plt.xlabel('Error',fontsize=18)
plt.ylabel('Effort',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)

plt.show()
# s_files = []
# data_files = []
# R1_values = []
# R2_values = []
# KL_values = []
#
# for s_list in file_names:
#     s_list = s_list.split('.png')
#     s_files.append(s_list[0])
#
# for i_list in s_files:
#     i_list = i_list.split('_')
#     data_files.append(i_list)
#     R1_values.append(float(i_list[1]))
#     R2_values.append(float(i_list[3]))
#     KL_values.append(float(i_list[5]))
#
# R1 = np.asarray(R1_values)
# R2 = np.asarray(R2_values)
# KL = np.asarray(KL_values)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
#
# R1_m = value_m(R1)
# R2_m = value_m(R2)

# A = scipy.io.loadmat('./data0/R1_0.1_R2_0.1_R3_0.1_KL_0.9491693423460088.mat')
# print(A['y1'])
