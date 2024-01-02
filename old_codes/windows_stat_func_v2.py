#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:36:45 2022

@author: erri
"""

import numpy as np
import os
run = 'q07_1'
DoD_name = 'DoD_s1-s0_filt_nozero_rst.txt'
home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')
DoD_path = os.path.join(DoDs_dir, 'DoD_' + run, DoD_name)
DoD = np.loadtxt(DoD_path, delimiter='\t')

array = np.where(DoD==-999, np.nan, DoD)

window_mode = 3
windows_length_base = 12


def windows_stat(array, window_mode, windows_length_base):
    '''
    This function provides statistics over different slicing methods of array.
    input:
        array: 2D numpy array with np.nans instead of -999
        windows_mode: integer
        windows_length_base: integer (in number of columns)
        plot_mode: integer
                    0: no plots
                    1: allows plots
    output:
        mean_array_tot : numpy array
                        array of data average for each window
        std_array_tot: numpy array
                        array of data standard deviation for each window
        x_data_tot: numpy array
                        array of windows dimension coherent with mean and std data
        window_boundary: numpy array
                        array of the windows boundary
    '''
    
    import numpy as np
    import os 
    import math
    W=windows_length_base
    mean_array_tot = []
    std_array_tot= []
    x_data_tot=[]
    window_boundary = np.array([0,0])
    
    if window_mode == 1:
        # With overlapping
        for w in range(1, int(math.floor(array.shape[1]/W))+1): # W*w is the dimension of every possible window   
            mean_array = []
            std_array= []
            x_data=[]
            for i in range(0, array.shape[1]+1):
                if i+w*W <= array.shape[1]:
                    window = array[:, i:W*w+i]
                    boundary = np.array([i,W*w+i])
                    window_boundary = np.vstack((window_boundary, boundary))
                    mean = np.nanmean(window)
                    std = np.nanstd(window)
                    mean_array = np.append(mean_array, mean)
                    std_array = np.append(std_array, std)
                    x_data=np.append(x_data, w)       
            mean_array_tot = np.append(mean_array_tot, np.nanmean(mean_array))
            std_array_tot= np.append(std_array_tot, np.nanstd(std_array)) #TODO check this
            x_data_tot=np.append(x_data_tot, np.nanmean(x_data))
            window_boundary = window_boundary[1,:]
    
    if window_mode == 2:
        # Without overlapping
        for w in range(1, int(math.floor(array.shape[1]/W))+1): # W*w is the dimension of every possible window
            mean_array = []
            std_array= []
            x_data=[]
            for i in range(0, array.shape[1]+1):
                if W*w*(i+1) <= array.shape[1]:
                    window = array[:, W*w*i:W*w*(i+1)]
                    boundary = np.array([W*w*i,W*w*(i+1)])
                    window_boundary = np.vstack((window_boundary, boundary))
                    mean = np.nanmean(window)
                    std = np.nanstd(window)
                    mean_array = np.append(mean_array, mean)
                    std_array = np.append(std_array, std)
                    x_data=np.append(x_data, w)
            mean_array_tot = np.append(mean_array_tot, np.nanmean(mean_array))
            std_array_tot= np.append(std_array_tot, np.nanstd(std_array)) #TODO check this
            x_data_tot=np.append(x_data_tot, np.nanmean(x_data))
            window_boundary = window_boundary[1,:]
    
    if window_mode == 3:
        # Increasing window dimension keeping still the upstream cross section
        mean_array = []
        std_array= []
        x_data=[]
        for i in range(0, array.shape[1]+1):
            if W*(i+1) <= array.shape[1]:
                window = array[:, 0:W*(i+1)]
                boundary = np.array([0,W*(i+1)])
                window_boundary = np.vstack((window_boundary, boundary))
                mean = np.nanmean(window)
                std = np.nanstd(window)
                mean_array = np.append(mean_array, mean)
                std_array = np.append(std_array, std)
                x_data=np.append(x_data, i)
        mean_array_tot = np.append(mean_array_tot, np.nanmean(mean_array))
        std_array_tot= np.append(std_array_tot, np.nanstd(std_array)) #TODO check this
        x_data_tot=np.append(x_data_tot, np.nanmean(x_data))
        window_boundary = window_boundary[1,:]

    return mean_array_tot, std_array_tot, x_data_tot, window_boundary

mean_array_tot, std_array_tot, x_data_tot, window_boundary = windows_stat(array, window_mode, windows_length_base)