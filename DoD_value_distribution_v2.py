#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:48:51 2023

@author: erri
"""

import numpy as np
import os
import matplotlib.pyplot as plt


runs = ['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q15_2', 'q20_2']
runs = ['q07_1', 'q10_2', 'q15_2', 'q20_2']
# runs = ['q20_2']
for timespan in range(10):
    
    timespan=0
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    for run,c in zip(runs, ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'p', 's']):
        home_dir = os.getcwd()
        report_dir = os.path.join(home_dir, 'output', run)
        plot_out_dir = os.path.join(home_dir, 'plot')
        
        data = np.load(os.path.join(home_dir,'output','DoDs', 'DoDs_stack', 'DoD_stack_'+run+'.npy'))
        
        
        
        data = data[:,:,:,timespan-1]
        
        '''
        DoD input stack structure:
            
            DoD_stack[time,y,x,delta]
            DoD_stack_bool[time,y,x,delta]
            
            - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >    delta
            |  DoD 1-0  DoD 2-0  DoD 3-0  DoD 4-0  DoD 5-0  DoD 6-0  DoD 7-0  DoD 8-0  DoD 9-0
            |  DoD 2-1  DoD 3-1  DoD 4-1  DoD 5-1  DoD 6-1  DoD 7-1  DoD 8-1  DoD 9-1
            |  DoD 3-2  DoD 4-2  DoD 5-2  DoD 6-2  DoD 7-2  DoD 8-2  DoD 9-2
            |  DoD 4-3  DoD 5-3  DoD 6-3  DoD 7-3  DoD 8-3  DoD 9-3
            |  DoD 5-4  DoD 6-4  DoD 7-4  DoD 8-4  DoD 9-4
            |  DoD 6-5  DoD 7-5  DoD 8-5  DoD 9-5
            |  DoD 7-6  DoD 8-6  DoD 9-6
            |  DoD 8-7  DoD 9-7
            |  DoD 9-8
            |
            v
            time
                
        '''
        
        
        
        # Choose the axis along which you want to unroll (flatten)
        axis_to_unroll = 0  # Change this to the desired axis (0, 1, or 2)
        
        # Use reshape to unroll along the specified axis
        unrolled_array = np.reshape(data, newshape=(-1, data.shape[axis_to_unroll]))
        unrolled_array = unrolled_array.T
        
        # Divide scour and deposition
        
        unrolled_array_sco = unrolled_array*(unrolled_array<0)
        unrolled_array_dep = unrolled_array*(unrolled_array>0)
        
        
        num_bins = 121
        hist_range = (-60, 60)  # Range of values to include in the histogram
        
        dist_array_tot = np.zeros((unrolled_array_sco.shape[0],num_bins))
        dist_array_sco = np.zeros((unrolled_array_sco.shape[0],num_bins))
        dist_array_dep = np.zeros((unrolled_array_dep.shape[0],num_bins))
        
        for t in range(unrolled_array.shape[0]):
            array_tot = unrolled_array[t,:]
            array_tot = array_tot[~np.isnan(array_tot)& (array_tot!=0)] # Trim np.nan
            hist, bin_edges = np.histogram(array_tot, bins=num_bins, range=hist_range)
            dist_array_tot[t,:] = hist/np.nansum(hist)
            
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
        dist_array_cum = np.cumsum(dist_array_mean)
        dist_array_stdev = np.std(dist_array, axis=0, ddof=0)
        dist_array_25perc = np.percentile(dist_array, 25, axis=0)
        dist_array_75perc = np.percentile(dist_array, 75, axis=0)
        
        dist_array_tot_mean = np.mean(dist_array_tot, axis=0)
        dist_array_tot_cum = np.cumsum(dist_array_tot_mean)
        
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
        