#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

Pixel age analysis over stack DoDs

INPUT (as .npy binary files):
    DoD_stack1 : 3D numpy array stack
        Stack on which DoDs are stored as they are, with np.nan
    DoD_stack1_bool : 3D numpy array stack
        Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
OUTPUTS:
    
    
"""
# Import
import os
import numpy as np
# import math
# import random
import time
import imageio
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat
# from matplotlib import colors
# from matplotlib.ticker import PercentFormatter
# import seaborn as sns

#%%
start = time.time() # Set initial time

# SCRIPT MODE
'''
run mode:
    0 = runs in the runs list
    1 = one run at time
    2 = bath process 'all the runs in the folder'
plot_mode:
    ==0 plot mode OFF
    ==1 plot mode ON
'''
run_mode = 1
plot_mode = 1

delta = 1 # Delta time of the DoDs


# ARRAY OF RUNS
# runs = ['q07_1', 'q10_2', 'q15_2', 'q15_3', 'q20_2']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q10_4','q15_2', 'q15_3', 'q20_2']
# runs = ['q10_3', 'q10_4']

runs = ['q07_1']
# runs = ['q10_2']
# runs = ['q15_2']
# runs = ['q15_3']
# runs = ['q20_2']


for run in runs:
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    run_dir = os.path.join(home_dir, 'surveys')
    DoDs_folder = os.path.join(home_dir, 'report', 'DoDs', 'DoDs_stack') # Input folder
    
    print('###############\n' + '#    ' + run + '    #' + '\n###############')
     
    #%%############################################################################
    '''
    This section calculate the envelop of an increasing number of 1-time-step DoD
    and extract the DoD in which the 1-time-step are contained:
    For example it calculate the envelope of DoD1-0, DoD2-1 and DoD3-2 and it also
    extract the DoD3-0.
    The envelop and the averall DoD maps were then translate as boolean activity map
    where 1 means activity and 0 means inactivity.
    Making the difference between this two maps (envelope - DoD), the diff map is a
    map of 1, -1 and 0.
    1 means that a pixel is active in the 1-time-step DoD evelope but is not detected
    as active in the overall DoD, so it is affcted by compensation processes.
    -1 means that a pixel is detected as active in the overall DoD but not in the
    1-time-step DoD so in the 1-time-step DoD it is always under the detection
    threshold (2mm for example).
    So this map provide us 2-dimensional information about pixel that experienced
    compensation or under thrashold processes.
    
    '''
    stack_dir = os.path.join(home_dir, 'output', 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    '''
    DoD input stack structure:
        
        DoD_stack[time,x,y,delta]
        DoD_stack_bool[time,x,y,delta]
        
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
    
    
    # INITIALIZE ARRAYS
    comp_array=[] # Initialize array where to append the number of pixel that experienced compensation
    thrs_array=[] # Initialize array ehere to append the number of pixel that experienced threshold bias
    
    comp_matrix = np.zeros(stack_bool[0,:,:,0].shape) # Compensation events matrix
    thrs_matrix = np.zeros(stack_bool[0,:,:,0].shape) # Under threshold events matrix
    
    # Define the report stack in which all the difference matrix will be stored
    report_stack = np.zeros((stack_bool.shape[0]-1,stack_bool.shape[1],stack_bool.shape[2],stack_bool.shape[3]-1))
    
    report_comp_matrix = np.zeros((stack_bool.shape[0]-1,stack_bool.shape[3]-1))
    report_thrs_matrix = np.zeros((stack_bool.shape[0]-1,stack_bool.shape[3]-1))
    index=0
    for i in range(0,stack.shape[3]-1): # For each delta
        
        cumul_comp_events_map = np.zeros(stack_bool[0,:,:,0].shape)
        cumul_thrs_eventss_map = np.zeros(stack_bool[0,:,:,0].shape)
        
        report_comp_array = []
        report_thrs_array = []
        
        for t in range(0,stack.shape[0]-1-index):
                    
            # print('DoD: ', i+1, t)
            # print('envMAA: ', t, t+2+i, 0)
            
            envMAA = np.nansum(np.abs(stack_bool[t:t+2+i,:,:,0]), axis=0) # envMAA
            envMAA = np.where(envMAA>0, 1, envMAA)
            
            DoD = abs(stack_bool[t,:,:,i+1])
            
            diff_matrix = envMAA - DoD
            
            comp_event_map = 1.0*(diff_matrix==1)
            thrs_event_map = 1.0*(diff_matrix==-1)
            
            cumul_comp_events_map += comp_event_map
            cumul_thrs_eventss_map += thrs_event_map
            
            # Reshape maps to get the right proportion of 1:10 in pixel dimension
            comp_event_map_rsh = np.repeat(comp_event_map, 10, axis=1)
            thrs_event_map_rsh = np.repeat(thrs_event_map, 10, axis=1)
            
            cumul_comp_events_map_rsh = np.repeat(cumul_comp_events_map, 10, axis=1)
            cumul_thrs_eventss_map_rsh = np.repeat(cumul_thrs_eventss_map, 10, axis=1)
            
            
            # Save compensation and threshold event spatially distributed as maps
            imageio.imwrite(os.path.join(report_dir, 'report_' + run, run + '_' + str(i) + '_' + str(t) + '_comp_events.tiff'), comp_event_map_rsh)
            imageio.imwrite(os.path.join(report_dir, 'report_' + run, run + '_' + str(i) + '_' + str(t) + '_thrs_events.tiff'), thrs_event_map_rsh)
            
            report_stack[i,:,:,t] = diff_matrix
            
            report_comp_array = np.append(report_comp_array, np.nansum(diff_matrix==1)/(stack_bool[0,:,:,0].shape[1]*120))
            report_thrs_array = np.append(report_thrs_array, np.nansum(diff_matrix==-1)/(stack_bool[0,:,:,0].shape[1]*120))
            # print(np.nansum(diff_matrix==1))
            # print(np.nansum(diff_matrix==-1))
            
            
            # print(len(report_comp_array))
            report_comp_matrix[i,0:len(report_comp_array)]  = report_comp_array
            report_thrs_matrix[i,0:len(report_thrs_array)] = report_thrs_array
            
        index += 1
        imageio.imwrite(os.path.join(report_dir, 'report_' + run, run + '_' + str(i) + '_cumul_comp_events.tiff'), cumul_comp_events_map_rsh)
        imageio.imwrite(os.path.join(report_dir, 'report_' + run, run + '_' + str(i) + '_cumul_thrs_events.tiff'), cumul_thrs_eventss_map_rsh)

    np.save(os.path.join(report_dir, 'report_' + run, 'comp_thrs_analysis', run + '_envMAA_DoD_difference.npy'), np.round(report_stack, decimals=4))
    np.savetxt(os.path.join(report_dir, 'report_' + run, 'comp_thrs_analysis', run + '_compensation_dimless_width.txt'), np.round(report_comp_matrix, decimals=4), delimiter=',')
    np.savetxt(os.path.join(report_dir, 'report_' + run, 'comp_thrs_analysis', run + '_under_thrs_dimless_width.txt'), np.round(report_thrs_matrix, decimals=4), delimiter=',')
    
    
end = time.time()
print()
print('Execution time: ', (end-start), 's')



#%%
# Manual check - for testing

DoD20 = abs(stack_bool[0,:,:,1])
DoD10 = abs(stack_bool[0,:,:,0])
DoD21 = abs(stack_bool[1,:,:,0])
envDoD20 = DoD10+DoD21
envDoD20 = np.where(envDoD20>0,1,envDoD20)
diff = envDoD20 - DoD20


# DoD30 = abs(stack_bool[0,:,:,2])
# DoD10 = abs(stack_bool[0,:,:,0])
# DoD21 = abs(stack_bool[1,:,:,0])
# DoD32 = abs(stack_bool[2,:,:,0])
# envDoD30 = DoD10+DoD21+DoD32
# envDoD30 = np.where(envDoD30>0,1,envDoD30)
# diff = envDoD30 - DoD30

# comp = np.nansum(diff==2) #/(278*120)
comp = np.nansum(diff==1) #/(278*120)
thrs = np.nansum(diff==-1) #/(278*120)

print('Compensation: ', comp)
print('Thrs: ', thrs)
