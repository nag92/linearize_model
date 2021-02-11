close all
clear all
clc

load('cost.mat')
J_hist = arr;
plot(arr)
xlabel('iteration')
ylabel('cost')
title('Cost varied with interations')
grid on