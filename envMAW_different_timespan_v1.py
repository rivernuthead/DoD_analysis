#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:40:06 2023

@author: erri

This script calculate the MAA envelope considering, to calculate it, DoDs
taken at different timespan. In particular this script calculate the MAW
envelope considering  using 8 DoDs taken at a timespan of 0.5 Txnr, 4 DoDs
taken at a timespan of 1.0 Txnr, 2 DoD's taken at a timespan of 2 Txnr and 
1 DoD taken at a timespan of 4 Txnr.
This would hepl me to understand the influence of the survey frequancy in the
computation of the MAW. 

"""


import os
import numpy as np
import matplotlib.pyplot as plt


# SINGLE RUN NAME
# RUNS = ['q15_3']
RUNS = ['q07_1', 'q10_2', 'q15_2', 'q15_3', 'q20_2']

# FOLDER SETUP
home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder

# Survey data
px, py = 50, 5 # [mm]


# COMPUTE THE MORPHOLOGICAL ACTIVE AREA

fig, ax = plt.subplots()


for run in RUNS:
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    # STACK DATA STRUCTURE
    # Create output matrix as below:
    #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
    # delta = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
    # delta = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
    # delta = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
    # delta = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
    # delta = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
    # delta = 6  6-0   7-1   8-2   9-3                                average STDEV
    # delta = 7  7-0   8-1   9-2                                      average STDEV
    # delta = 8  8-0   9-1                                            average STDEV
    # delta = 9  9-0                                                  average STDEV
    
    # stack[time, x, y, delta]
    
    timespan = np.array([1,2,4,8])
    delta = np.array([8,4,2,1])
    envMAW = []
    envMAV = []
    for ts, d in zip(timespan, delta):
        # print(ts, d)
        # Envelope in terms of activity (1=active, 0=inactive)
        envelope_activity = np.nansum(abs(stack_bool[:d, :,:,ts-1]), axis=0)
        envelope_activity = np.where(envelope_activity!=0, 1, 0)
        envMAW = np.append(envMAW, np.nansum(envelope_activity)/(dim_x*dim_y))
        
        # Envelope in terms of volumes
        envelope_volumes = np.nansum(abs(stack[:d, :,:,ts-1]), axis=0)
        envMAV = np.append(envMAV, np.nansum(envelope_volumes)*dim_x*dim_y)
    
    np.savetxt(os.path.join(report_dir, run,run+ 'envMAW_different_timespan.txt'), envMAW, delimiter=',')
    np.savetxt(os.path.join(report_dir, run,run+ 'envMAV_different_timespan.txt'), envMAV, delimiter=',')
        
    
    
    #Plot

    ax.plot(timespan/2, envMAW, label='envMAW_'+run, marker='o', markersize=5, linestyle= '-')
    
    # # Shows dava values near the plot dots
    # for i, value in enumerate(envMAW):
    #     ax.text(timespan[i], value, f'{value:.2f}', fontsize=8, ha='center', va='bottom')
    
# add labels, title, and legend
ax.set_xlabel('Timespan')
ax.set_ylabel('dimensionless envMAW ')
ax.set_title('Value of MAW envelope at different timespan')
ax.legend()

# add some additional formatting
ax.grid(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(report_dir, 'envMAW_different_timespan.pdf'), dpi=300)
plt.savefig(os.path.join(report_dir, 'envMAW_different_timespan.png'), dpi=300)

# show the plot
plt.show()







# COMPUTE THE MORPHOLOGICAL ACTIVE VOLUMES
fig, ax = plt.subplots()


for run in RUNS:
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    # STACK DATA STRUCTURE
    # Create output matrix as below:
    #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
    # delta = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
    # delta = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
    # delta = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
    # delta = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
    # delta = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
    # delta = 6  6-0   7-1   8-2   9-3                                average STDEV
    # delta = 7  7-0   8-1   9-2                                      average STDEV
    # delta = 8  8-0   9-1                                            average STDEV
    # delta = 9  9-0                                                  average STDEV
    
    # stack[time, x, y, delta]
    
    timespan = np.array([1,2,4,8])
    delta = np.array([8,4,2,1])
    # envMAW = []
    envMAV_fill = []
    envMAV_sco = []
    for ts, d in zip(timespan, delta):
        # print(ts, d)
        # Envelope in terms of volumes of fill
        # TODO this has to be checked!
        
        envelope_volumes_fill = np.nansum(abs(stack[:d, :,:,ts-1]*(stack[:d, :,:,ts-1]>0)), axis=0)
        envMAV_fill = np.append(envMAV_fill, np.nansum(envelope_volumes_fill)*px*py/1e06)
        
        # Envelope in terms of volumes of sco
        envelope_volumes_sco = np.nansum(abs(stack[:d, :,:,ts-1]*(stack[:d, :,:,ts-1]<0)), axis=0)
        envMAV_sco = np.append(envMAV_sco, np.nansum(envelope_volumes_sco)*px*py/1e06)
    
    # np.savetxt(os.path.join(report_dir, run,run+ 'envMAW_different_timespan.txt'), envMAW, delimiter=',')
    np.savetxt(os.path.join(report_dir, run,run+ 'envMAV_fill_different_timespan.txt'), envMAV_fill, delimiter=',')
    np.savetxt(os.path.join(report_dir, run,run+ 'envMAV_sco_different_timespan.txt'), envMAV_sco, delimiter=',')
        
    
    
    #Plot

    ax.plot(timespan/2, envMAV_fill, label='envMAV_'+run, marker='o', markersize=5, linestyle= '-')
    
    # # Shows dava values near the plot dots
    # for i, value in enumerate(envMAW):
    #     ax.text(timespan[i], value, f'{value:.2f}', fontsize=8, ha='center', va='bottom')
    
# add labels, title, and legend
ax.set_xlabel('Timespan')
ax.set_ylabel('Morphological Active Volumes [dm³]')
ax.set_title('Value of fill MAV envelope at different timespan (Fill)')
ax.legend()

# add some additional formatting
ax.grid(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(report_dir, 'envMAV_fill_different_timespan.pdf'), dpi=300)
plt.savefig(os.path.join(report_dir, 'envMAV_fill_different_timespan.png'), dpi=300)

# show the plot
plt.show()