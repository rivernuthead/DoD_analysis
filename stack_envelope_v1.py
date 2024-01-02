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
# import matplotlib.pyplot as plt
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
run_mode = 0
plot_mode = 1

delta = 1 # Delta time of the DoDs

# # SINGLE RUN NAME
# run = 'q15_3'
# ARRAY OF RUNS
RUNS = ['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q15_3', 'q20_2']

# FOLDER SETUP
home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder

for run in RUNS:
    # # Create the run name list
    # RUNS=[]
    # if run_mode==0:
    #     RUNS=runs
    # elif run_mode ==2: # batch run mode
    #     for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
    #         if RUN.startswith('q'): # Consider only folder names starting wit q
    #             RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
    # elif run_mode==1: # Single run mode
    #     RUNS=run.split() # RUNS as a single entry array, provided by run variable
    # elif run_mode==3:
    #     # RUNS=['q10_2', 'q10_3', 'q10_4', 'q10_5', 'q10_6']
    #     RUNS=['q07_1, q10_2', 'q10_3', 'q10_7', 'q15_2', 'q20_2']
    print('###############\n' + '#    ' + run + '    #' + '\n###############')
    
    
    
    
    # Step between surveys
    
    ###############################################################################
    # IMPORT RUN PARAMETERS from file parameters.txt
    # variable run must be as 'q' + discharge + '_' repetition number
    # Parameters.txt structure:
    # discharge [l/s],repetition,run time [min],Texner discretization [-], Channel width [m], slope [m/m]
    # Load parameter matrix:
    parameters = np.loadtxt(os.path.join(home_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)
    # Extract run parameter depending by run name
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]
    
    # Run time data
    dt = run_param[0,2] # dt between runs [min] (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)


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
1 means that a pixel is active in the 1-time-step DoD but is not detected
as active in the overall DoD, so it is affcted by compensation processes.
-1 means that a pixel is detected as active in the overall DoD but not in the
1-time-step DoD so in the 1-time-step DoD it is always under the detection
threshold (2mm for example).
So this map provide us 2-dimensional information about pixel that experienced
depositional or under thrashold processes.

'''

for run in RUNS:
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    
    
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
    
    
    # for t in range(0, stack.shape[0]-1):
    for d in range(0, stack.shape[0]-1):
        comp_array=[] # Initialize array where to append the number of pixel that experienced compensation
        thrs_array=[] # Initialize array ehere to append the number of pixel that experienced threshold bias
        for t in range(0, stack.shape[0]-1):
            DoD_envelope = np.nansum(abs(stack_bool[:t+2,:,:,d]), axis=0) # Perform the DoD envelope
            DoD_envelope = np.where(DoD_envelope>0, 1, DoD_envelope) # Make the envelope of DoD as a boolean map
            DoD_boundary = abs(stack_bool[0,:,:,t+1]) # Extract the DoD within at which the envelope take place
            diff_matrix = DoD_envelope - DoD_boundary # Perform the difference between the DoDs_envelope and the boundary DoD
            comp = np.nansum(diff_matrix==1) # Compute the number of pixel that experience compensation
            thrs = np.nansum(diff_matrix==-1) # Compute the number of pixel that experienced threshold bias
            comp_array = np.append(comp_array, comp) # Append values into the array
            thrs_array = np.append(thrs_array, thrs) # Append values into the array
        # Save array in txt
        np.savetxt(os.path.join(report_dir, run, run + '_' + str(d+1) + '_comp_analysis.txt'), comp_array, delimiter='\t')
        np.savetxt(os.path.join(report_dir, run, run + '_' + str(d+1) + '_thrs_analysis.txt'), thrs_array, delimiter='\t')





#%%
###############################################################################
'''

'''

for run in RUNS:
    # Initialize variables
    envelope_t_increment_array=[]
    
    #Import data
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    for t in range(0, stack_bool.shape[0]):
        envelope_t = np.nansum(abs(stack_bool[:t,:,:,0]), axis=0) # This is the envelope of the first t 1-time-step DoD
        envelope_t = np.where(envelope_t>0, 1, envelope_t) # Set the envelope as a boolean map of activity and inactivity
        envelope_t_area = np.nansum(envelope_t[:,:]) # Compute the active area
        
        
        envelope_t1 = np.nansum(abs(stack_bool[:t+1,:,:,0]), axis=0) # This is the envelope of the first t+1 1-time-step DoD
        envelope_t1 = np.where(envelope_t1>0, 1, envelope_t1) # Set the envelope as a boolean map of activity and inactivity
        envelope_t1_area = np.nansum(envelope_t1[:,:]) # Compute the active area
        
        
        envelope_t_increment = envelope_t1_area-envelope_t_area # Compute the difference between the two envelope
        
        envelope_t_increment_array = np.append(envelope_t_increment_array, envelope_t_increment) # Append increment value
    np.savetxt(os.path.join(report_dir, run, run + '_envelope_increment.txt'), envelope_t_increment_array, delimiter=',')

#%%
###############################################################################
'''
MAW envelope considering an increasing number of DoDs
'''
delta=0

for run in RUNS:
    # Initialize variables
    envelope_area_array=[]
    
    #Import data
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    for t in range(1, stack_bool.shape[0]+1):
        envelope = np.nansum(abs(stack_bool[:t,:,:,delta]), axis=0) # This is the envelope of the first t 1-time-step DoD
        envelope = np.where(envelope>0, 1, envelope) # Set the envelope as a boolean map of activity and inactivity
        envelope_area = np.nansum(envelope[:,:]) # Compute the active area
        
        envelope_area_array = np.append(envelope_area_array, envelope_area*0.005*0.05/(stack_bool.shape[2]*0.05)/0.6) # Append increment value
    
    np.savetxt(os.path.join(report_dir, run, run + '_envMAW.txt'), envelope_area_array, delimiter=',')

#%%
###############################################################################
'''
Envelope calculation for different delta and pixel that have experienced a switch
'''

# INITIALIZE STACK AND ARRAY
act_time_stack = np.zeros(stack.shape[:3]) # activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
switch_matrix = np.zeros(stack.shape[1:3]) # This is the 2D matrix that collect the number of switch over time


delta=0

for run in RUNS:
    # Initialize variables
    envelope_area_array=[]
    
    #Import data
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
    for t in range(1, stack_bool.shape[0]+1):
        # Envelope computing
        envelope = np.nansum(abs(stack_bool[:t,:,:,delta]), axis=0) # This is the envelope of the first t 1-time-step DoD
        envelope = np.where(envelope>0, 1, envelope) # Set the envelope as a boolean map of activity and inactivity
        envelope_area = np.nansum(envelope[:,:]) # Compute the active area
        
        envelope_area_array = np.append(envelope_area_array, envelope_area*0.005*0.05/(stack_bool.shape[2]*0.05)/0.6) # Append increment value
        
        
        for x in range(0,stack_bool.shape[2]-2):
            for y in range(0,stack_bool.shape[1]-1):
                slice_array = stack_bool[:,y,x,0] # Slice the stack in a single pixel array where data is collected over time
                # slice_array = np.array([1., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.])
                # slice_array = np.array([0,0,0,0,0,-1,-1,0,0,0,1,0,0,0,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,1,0,0,0]) # Test array
                # slice_array = np.array([0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0]) # Test array
                n_zero = 0 # This is the number of zero values before the first non-zero value in the sliced array. It is initialized to zero.

                # Check if the sliced array has np.nan. If so, fill this array with np.nan and then fill the stack matrix
                if np.isnan(slice_array).any(): # check if a has np.nan value, if so fill matrix with np.nan
                    switch_matrix[y,x] = np.nan
                    act_time_stack[:,y,x] = np.nan
                    pass
                
                # This part checks if there are zero values before the first non-zero value in he sliced array. If there are at least one, the value of n-zero variable will be update with the numbero of zeros       
                if slice_array[0]==0:# If the first entry of the sliced array is 0, in this schema it could have both scour and deposition nature.
                # So the length and the nature of the first period depend by the nature of the first non-zero value.
                    if np.sum(slice_array==0) == len(slice_array): # Check if the sliced array is full of zeros
                        time_array = np.array([np.nan]) # Fill the array with np.nan to keep them transparent
                    else:
                        n_zero=np.array(np.where(slice_array!=0))[0,0] # Number of zero values before the first non-zero value
                        
                        slice_array=slice_array[n_zero:] # Trim the zero values before the first non-zero value.
                        
                time_array = []
                if slice_array[0]!=0: # If the first entry of the sliced array is non-zero
                    count=1 # Initialize the count variable. This variable will count the number of activation instants
                    target_sign = np.sign(slice_array[0]) # This variable collects the sign of the first element of each same-nature period
                    for i in range(0,len(slice_array)-1): # Loop over the sliced array
                        a1, a2 = slice_array[i], slice_array[i+1] # a1 and a2 are the two adjacent element in the sliced array
                        if np.sign(a1)==np.sign(a2): # If two consecutive elements have the same naure
                            count += 1 # If two consecutive elements have the same nature the count increases
                        elif np.sign(a1)*np.sign(a2)==0 and (np.sign(a2)==target_sign or np.sign(a1)==target_sign):
                            count += 1 # The count increases also if one or both elements are zero but the non-zero value has the target sign
                        elif np.sign(a1)!=np.sign(a2) and (np.sign(a2)!=target_sign or np.sign(a1)!=target_sign): # The count stops when a switch occours or when one of the two elements shows a sign different from the target sign
                            time_array = np.append(time_array, count*target_sign)  # This operation append to time_array the count value with his sign. This could be useful to keep trace of the nature of the period.
                            target_sign=-1*target_sign # Update the target sign
                            count=1 # Update the count variable that will starts again from zero
                            pass
                            
                    time_array = np.append(time_array, (len(slice_array)-np.sum(np.abs(time_array)))*target_sign) # By now the last period is not calculated (actually because, as the first one, it is only a lower boundary of time because it doesn't appear within two switches) so this operation appeds this value manually
                    # time_array[0] = time_array[0] + np.sign(time_array[0])*n_zero # Ths operation append, if present, the number of zeroes before the first non-zero value calculated on the very first sliced array (n_zero variable)
                
                ind = np.max(np.where(time_array!=0)) # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
                # So in the filling process I want to exclude the last period:
                act_time_stack[:ind,y,x]=time_array[:ind] # This operation fills the stack with time_array     


                if len(time_array)==0 or len(time_array)==1: # If sliced array does not contain any switch (so if the length of the time array is 0 in the case we do not consider the last period - see above - or if the length is 1 so only one period is considered - the last one - this operation fills the output matrix with np.nan)
                    # switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
                    for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                        act_time_stack[t,y,x]=np.nan
                
                # elif len(time_array)==2: # If only one switch occours, two periods were identified. This two periods are not completely reliable since they were not observed between two switches
                #     switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
                #     for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                #         act_time_stack[t,y,x]=np.nan
                    
                else: # If at least one (or two) switches occours
                    # Fill switch matrix
                    switch_matrix[y,x] = len(time_array[time_array!=0]) # To provide the number of switch we decide to trim the zero values to keep only the number of switch
                    
                    # Fill activation time stack
                    # act_time_stack[:len(time_array),y,x]=time_array # To provide the time between each detected switch
                    ind = np.max(np.where(time_array!=0)) # This number correspond to the index of the last period in the time_array that is not reliable (as the first one)
                    # So in the filling process I want to exclude the last period:
                    act_time_stack[:ind,y,x]=time_array[:ind] # To provide the time between each detected switch


        '''
        Period matrix calculation
        This matrix collect, for each pixels, the numer of active periods.
        This periods are at least 1, if the nature has not changed during the observations.
        Otherwise these periods could be a maximum of the number of observation if the
        nature of the pixel has changed every time.
        The switches matrix will born from the periods_matrix applyng the following obvious operation
        number of switch = number of periods - 1
        '''
        # From the act_time_stack, switch_matrix will be created as a matrix where each period will be converted as a 1
        periods_matrix = np.where(act_time_stack!=0,1,0)*np.where(np.isnan(act_time_stack),np.nan,1) # The multiplication is needed to apply the np.nan mask
        # So, this 3D matrix will be converted as a 2D metrix by a sum operation
        periods_matrix = np.sum(periods_matrix, axis=0)

        switches_matrix = periods_matrix-1

        
    np.savetxt(os.path.join(report_dir, run, run + '_envelope_area.txt'), envelope_area_array, delimiter=',')
#%%
end = time.time()
print()
print('Execution time: ', (end-start), 's')