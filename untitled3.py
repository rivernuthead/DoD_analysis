#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:56:12 2023

@author: erri

"""


import numpy as np
import os
import matplotlib.pyplot as plt


script_name = os.path.basename(__file__)
print(script_name)

# runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']

# runs = ['q07_1', 'q10_2', 'q15_2']


runs = ['q20_2']
runs = ['q15_2']
runs = ['q10_2']
runs = ['q07_1']

delta = 1

# num_bins = 11
# bin_edges = [-40.0,-10.0,-8.0,-6.0,-4.0,-2.0,2.0,4.0,6.0,8.0,10.0,40.0]
# hist_range = (-40, 40)  # Range of values to include in the histogram

for run in runs:
    home_dir = os.getcwd()
    report_dir = os.path.join(home_dir, 'output', 'report_' + run)
    output_dir = os.path.join(report_dir, 'dist_step_by_step')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    plot_out_dir = os.path.join(home_dir, 'plot')
    
    data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
    
    timespan = 1
    
    data = data[:,:,:,timespan-1]
    
    
    # Step 1: Unroll the data
    unrolled_data = data.flatten()
    mask = unrolled_data != 0
    unrolled_data = unrolled_data[mask]
    unrolled_data = unrolled_data[~np.isnan(unrolled_data)]
    
    # num_bins = 120
    
    bin_edges = np.linspace(-60,60,601)
    
    # Compute the histogram
    hist, bin_edges = np.histogram(unrolled_data, bins=bin_edges)
    
    hist = hist/np.nansum(hist)

    cumulative_sum = np.cumsum(hist)
    
    