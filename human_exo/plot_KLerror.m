clear all
close all
clc

load('error.mat')

R1_values = arr1;
R2_values = arr2;
KL_values = arr3;

%plot3(R1_values,R2_values,KL_values,'o')

X = [R1_values;R2_values];
Weight = KL_values;

G = [X',Weight'];

hist3(X')