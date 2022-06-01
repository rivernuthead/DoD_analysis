#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:46:32 2022

@author: erri
"""
    
import numpy as np
import os 
import math
DoD_path = '/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/DoDs/DoD_q07_1/DoD_s1-s0_filt_nozero_rst.txt'
DoD = np.loadtxt(DoD_path, delimiter='\t')

DoD = np.where(DoD==-999, np.nan, DoD)

W = 12


mean_array = []
std_array= []
x_data=[]
mean_array_tot = []
std_array_tot= []
x_data_tot=[]


# With overlapping
for w in range(W, int(math.floor(DoD.shape[1]/W))):
    for i in range(0, DoD.shape[1]):
        if i+w <= DoD.shape[1]:
            window = DoD[:, i:i+w]
            mean = np.nanmean(window)
            std = np.nanstd(window)
            mean_array = np.append(mean_array, mean)
            std_array = np.append(std_array, std)
            x_data=np.append(x_data, w)
            
    mean_array_tot = np.append(mean_array_tot,)
    std_array_tot= []
    x_data_tot=[]


# Without overlapping
for n in range(1, int(math.floor(DoD.shape[1]/W))):
    w = W*n # Windows analysis dimension
    # print(w)
    for i in range(0, DoD.shape[1]):
        if w*(i+1) <= DoD.shape[1]:
            print(i*w, (i+1)*w)
            window = DoD[:, w*i:w*(i+1)]
            mean = np.nanmean(window)
            std = np.nanstd(window)
            mean_array = np.append(mean_array, mean)
            std_array = np.append(std_array, std)
            x_data=np.append(x_data, w)
        