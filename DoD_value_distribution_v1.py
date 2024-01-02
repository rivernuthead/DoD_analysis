#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:48:51 2023

@author: erri
"""

import numpy as np
import os
import matplotlib.pyplot as plt


runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']
timespan = 10 # This is the timespan at which every DoD was taken

for timespan in range(10):

    
    # Create the figure and axis
    fig, ax = plt.subplots()
    for run,c in zip(runs, ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'p', 's']):
        home_dir = os.getcwd()
        report_dir = os.path.join(home_dir, 'output', run)
        plot_out_dir = os.path.join(home_dir, 'plot')
        
        data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
        
        
        
        data = data[:,:,:,timespan-1]
        
        # Create output matrix as below:
        #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
        # timespan = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
        # timespan = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
        # timespan = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
        # timespan = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
        # timespan = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
        # timespan = 6  6-0   7-1   8-2   9-3                                average STDEV
        # timespan = 7  7-0   8-1   9-2                                      average STDEV
        # timespan = 8  8-0   9-1                                            average STDEV
        # timespan = 9  9-0                                                  average STDEV
        
        # stack[time, x, y, timespan]
        
        
        
        # Choose the axis along which you want to unroll (flatten)
        axis_to_unroll = 0  # Change this to the desired axis (0, 1, or 2)
        
        # Use reshape to unroll along the specified axis
        unrolled_array = np.reshape(data, newshape=(-1, data.shape[axis_to_unroll]))
        unrolled_array = unrolled_array.T
        
        # Divide scour and deposition
        
        unrolled_array_sco = unrolled_array*(unrolled_array<0)
        unrolled_array_dep = unrolled_array*(unrolled_array>0)
        
        
        num_bins = 120
        hist_range = (-40, 40)  # Range of values to include in the histogram
        
        dist_array_sco = np.zeros((unrolled_array_sco.shape[0],num_bins))
        dist_array_dep = np.zeros((unrolled_array_dep.shape[0],num_bins))
        
        for t in range(unrolled_array.shape[0]):
            array_sco = unrolled_array_sco[t,:]
            array_sco = array_sco [~np.isnan(array_sco) & (array_sco!=0)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(array_sco, bins=num_bins, range=hist_range)
            dist_array_sco[t,:] = hist/np.nansum(hist)
            
            array_dep = unrolled_array_dep[t,:]
            array_dep = array_dep [~np.isnan(array_dep) & (array_dep!=0)] # Trim 0 and np.nan
            hist, bin_edges = np.histogram(array_dep, bins=num_bins, range=hist_range)
            dist_array_dep[t,:] = hist/np.nansum(hist)
        
        dist_array = dist_array_sco + dist_array_dep
        
        dist_array_mean = np.mean(dist_array, axis=0)
        dist_array_stdev = np.std(dist_array, axis=0, ddof=0)
        dist_array_25perc = np.percentile(dist_array, 25, axis=0)
        dist_array_75perc = np.percentile(dist_array, 75, axis=0)
            
        
        # Plot the distributions

        x_data = np.linspace(hist_range[0], hist_range[1], num_bins)
        
        # Plot the mean as a line
        ax.plot(x_data , dist_array_mean, label=run, color=c, linewidth=0.5    )
        
        # Plot the variance as a colored area
        # ax.fill_between(x_data, dist_array_mean - dist_array_stdev, dist_array_mean + dist_array_stdev, alpha=0.5, label='Variance', color='g')
        
        # Plot the variance as a colored area
        ax.fill_between(x_data, dist_array_25perc, dist_array_75perc, alpha=0.5, label='Variance', color=c, edgecolor='none')
        
        
    # Customize the plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.ylim(0,0.18)
    plt.title('Mean and Variance Plot - timespan='+str(timespan))
    plt.legend()
    plt.grid(True)
        
    # Save the figure as a PDF with a specified DPI
    plt.savefig(os.path.join(plot_out_dir, 'DoDs_value_DoF - timespan'+ str(timespan)+'.pdf'), format='pdf', dpi=300)  # Specify DPI (e.g., 300)
    
    # Show the plot
    plt.show()
        