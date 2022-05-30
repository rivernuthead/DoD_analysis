#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:58:16 2022

@author: erri

Function to define lover and upper limits of the slicing windows of a moving
average and stdev analysis with increasing domain
"""

import numpy as np
import math

DoD_path = '/home/erri/Documents/Research/5_research_repos/DoD_analysis/DoDs/DoD_q20_2/DoD_s9-s8_filt_nozero_rst.txt'
DoD_filt_nozero = np.loadtxt(DoD_path, delimiter='\t')
analysis_window = np.array([12, 24])


'''
input:
    DoD matrix
        Numpy array 1D 2D
    Analisys window
        Numpy array of int
    overlapping_mode: int
        0 = no overlapping
        1 = overlapping

output:
    window_boundary
        Numpy array of lower and upper limit for each possible windows
    
'''

dim_x = DoD_filt_nozero.shape[1]
boundary_windows = []
for m in analysis_window: # m is the window dimension


    if overlapping_mode == 0:
        lower_lim = 
    
