#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy.io
import scipy.interpolate

def value_m(M_in):
    M_out = np.zeros(len(M_in))
    for i in range(len(M_in)):
        if M_in[i] > 9.8e-4:
            M_out[i] = M_in[i]/10 + 9e-4
        else:
            M_out[i] = M_in[i]
    return M_out


file_names = os.listdir('./figures_new')
s_files = []
data_files = []
R1_values = []
R2_values = []
KL_values = []

for s_list in file_names:
    s_list = s_list.split('.png')
    s_files.append(s_list[0])

for i_list in s_files:
    i_list = i_list.split('_')
    data_files.append(i_list)
    R1_values.append(float(i_list[1]))
    R2_values.append(float(i_list[3]))
    KL_values.append(float(i_list[5]))

R1 = np.asarray(R1_values)
R2 = np.asarray(R2_values)
KL = np.asarray(KL_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

R1_m = value_m(R1)
R2_m = value_m(R2)


#xi, yi = np.meshgrid(R1,R2)
# x, y = R1.ravel(), R2.ravel()
#
# z = np.zeros(len(x))
# for i in range(1,len(x)):
#     z[i] = KL_values[int((i*len(KL))/len(x))]
#
# bot = np.zeros_like(z)
# width = depth = 1e-2
#
# ax.bar3d(x,y,KL_values,bot,width,depth)




# ssscipy.io.savemat('error.mat',mdict={'arr1':R1, 'arr2':R2, 'arr3':KL})



#x = s[0].split('_')

### split('_')

### os.listdir

# xi = R1
# yi = R2
# zi = scipy.interpolate.griddata((R1,R2),KL,(xi[None,:], yi[:,None]),method='cubic')
#
# xi,yi = np.meshgrid(xi,yi)
#
# surf = ax.plot_surface(xi[1:-1,1:-1], yi[1:-1,1:-1],zi[1:-1,1:-1])
# plt.show()

# x = R1_m
# y = R2_m
# z = np.zeros(len(KL))
# dx = np.ones(len(R1_m))*1e-4
# dy = np.ones(len(R1_m))*1e-4
# dz = KL
#
# #ax.bar3d(x,y,z,dx,dy,dz,shade='True')
#
# for i in range(len(KL)):
#     color = np.array([1,1,1-dz[i]/max(KL)])
#     ax.bar3d(x[i],y[i],z[i],dx[i],dy[i],dz[i],color,shade='true')
#     # if dz[i] <= 40:
#     #     ax.bar3d(x[i],y[i],z[i],dx[i],dy[i],dz[i],'r',shade='true')
#     # else:
#     #     color = np.array([1,1,1-dz[i]/max(KL)])
#     #     ax.bar3d(x[i],y[i],z[i],dx[i],dy[i],dz[i],color,shade='true')
#
# #ax.bar3d(x,y,z,dx,dy,dz,shade='True')
#
# # R_xy = ['0.0001','0.0002','0.0003','0.0004','0.0005','0.0006','0.0007','0.0008','0.0009','0.001','0.002','0.003','0.004','0.005','0.006','0.007','0.008','0.009']
# # R_origin = [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,10e-4,11e-4,12e-4,13e-4,14e-4,15e-4,16e-4,17e-4,18e-4]
# R_xy = ['1','5','10','50','90']
# R_origin = [1e-4,5e-4,10e-4,14e-4,18e-4]
# ax.set_xlabel('R values of the left leg ($10^{-4}$)',fontsize=18)
# ax.set_xticks(R_origin)
# ax.set_xticklabels(R_xy,fontsize=15)
# ax.set_ylabel('R values of the right leg ($10^{-4}$)',fontsize=18)
# ax.set_yticks(R_origin)
# ax.set_yticklabels(R_xy,fontsize=15)
# ax.set_zlabel('Errors',fontsize=20)
# ax.set_title('Plotting trajectory errors',fontsize=30)

Xs = R1_m
Ys = R2_m
Zs = KL/176.0
surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

# R_xy = ['0.0001','0.0002','0.0003','0.0004','0.0005','0.0006','0.0007','0.0008','0.0009','0.001','0.002','0.003','0.004','0.005','0.006','0.007','0.008','0.009']
# R_origin = [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,10e-4,11e-4,12e-4,13e-4,14e-4,15e-4,16e-4,17e-4,18e-4]
R_xy = ['1','5','10','50','90']
R_origin = [1e-4,5e-4,10e-4,14e-4,18e-4]
ax.set_xlabel('R values of the left leg ($10^{-4}$)',fontsize=18)
ax.set_xticks(R_origin)
ax.set_xticklabels(R_xy,fontsize=15)
ax.set_ylabel('R values of the right leg ($10^{-4}$)',fontsize=18)
ax.set_yticks(R_origin)
ax.set_yticklabels(R_xy,fontsize=15)
ax.set_zlabel('Errors',fontsize=20)
ax.set_title('Plotting trajectory errors',fontsize=30)
