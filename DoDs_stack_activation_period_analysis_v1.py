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
import math
import random
import time
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns

#%%
start = time.time() # Set initial time

# SCRIPT MODE
'''
plot_mode:
    ==0 plot mode OFF
    ==1 plot mode ON
'''
plot_mode = 1

delta = 1 # Delta time of the DoDs

# SINGLE RUN NAME
run = 'q20_2'
run_mode = 3

# FOLDER SETUP
home_dir = os.getcwd() # Home directory
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder


# Create the run name list
RUNS=[]
if run_mode ==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run.split() # RUNS as a single entry array, provided by run variable
elif run_mode==3:
    # RUNS=['q10_2', 'q10_3', 'q10_4', 'q10_5', 'q10_6']
    RUNS=['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q10_7', 'q15_2', 'q20_2']
    
print('###############\n' + '#    ' + run + '    #' + '\n###############')


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

###############################################################################
# IMPORT doD STACK AND DoD BOOL STACK
stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path

stack = np.load(stack_path) # Load DoDs stack
stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack


stack = stack[:,:,:,delta-1]
stack_bool = stack_bool[:,:,:,delta-1]

dim_t, dim_y, dim_x = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension

#%%##############################################################################
# INITIALIZE STACK AND ARRAY
act_time_stack = np.zeros(stack.shape) # activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
switch_matrix = np.zeros(stack.shape[1:]) # This is the 2D matrix that collect the number of switch over time


#%%
'''
Activation time counting:
This loop works all over the x and y spatial coordinates calculating the duration of each activation period.

input:
    stack_bool as numpy stack dimension(time,dim_y,dim_x) where (1=dep, 0=no_changes, -1=sco)

output:
    act_time_stack as numpy stack dimension(time,dim_y,dim_x) where the number stored is the duration in
        timesteps of each active period
'''

for x in range(0,dim_x):
    for y in range(0,dim_y):
        slice_array = stack_bool[:,y,x] # Slice the stack in a single pixel array where data is collected over time
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
np.savetxt(os.path.join(report_dir, run, run + 'switch_matrix.txt'), switches_matrix, delimiter=',')


#%%############################################################################
'''
Activation time counting:
This loop works all over the x and y spatial coordinates calculating the duration of each activation period.

input:
    stack_bool as numpy stack dimension(time,dim_y,dim_x) where (1=dep, 0=no_changes, -1=sco)

output:
    act_time_stack as numpy stack dimension(time,dim_y,dim_x) where the number stored is the duration in
        timesteps of each active period
'''


stack = np.load(stack_path) # Load DoDs stack
stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack


stack = stack[:,:,:,delta-1]
stack_bool = stack_bool[:,:,:,delta-1]

for t in range(2,stack_bool.shape[0]):
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack


    stack = stack[:t,:,:,delta-1]
    stack_bool = stack_bool[:t,:,:,delta-1]
    
    dim_t, dim_y, dim_x = stack.shape[:3] # Define time dimension, crosswise dimension and longitudinal dimension
    
    # INITIALIZE STACK AND ARRAY
    act_time_stack = np.zeros(stack.shape) # activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
    switch_matrix = np.zeros(stack.shape[1:]) # This is the 2D matrix that collect the number of switch over time
    
    for x in range(0,dim_x):
        for y in range(0,dim_y):
            slice_array = stack_bool[:,y,x] # Slice the stack in a single pixel array where data is collected over time
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
    np.savetxt(os.path.join(report_dir, run, run + '_' + str(t) + 'switch_matrix.txt'), switches_matrix, delimiter=',')


#%%############################################################################
'''
Count the number of consecutive scour, deposition and zeros in the act_time_stack 
'''
stack_bool_sco = stack_bool<0
stack_bool_dep = stack_bool>0
stack_bool_zeros = stack_bool==0


stack_bool_sco = np.where(stack_bool_sco==1, 1, np.nan)
stack_bool_dep = np.where(stack_bool_dep==1, 1, np.nan)
stack_bool_zeros = np.where(stack_bool_zeros==1, 1, np.nan)

matrix_dep = np.zeros(act_time_stack.shape)
matrix_sco = np.zeros(act_time_stack.shape)
matrix_zeros = np.zeros(act_time_stack.shape)


# yyy = 10
# xxx = 11


for i in range(0,act_time_stack.shape[1]): #act_time_stack.shape[1]
    for j in range(0,act_time_stack.shape[2]): #act_time_stack[2]
# i=yyy
# j=xxx
        dep_count=0
        sco_count=0
        zeros_count=0
        for t in range(0,act_time_stack.shape[0]-1): #act_time_stack.shape[0]+1
        
            if stack_bool_dep[t,i,j]==stack_bool_dep[t+1,i,j]:
                dep_count+=1
            elif stack_bool_dep[t,i,j]!=stack_bool_dep[t+1,i,j] and not(np.isnan(stack_bool_dep[t,i,j])) :
                matrix_dep[t,i,j]=dep_count+1
                dep_count=0
                
            if stack_bool_sco[t,i,j]==stack_bool_sco[t+1,i,j]:
                sco_count+=1
            elif stack_bool_sco[t,i,j]!=stack_bool_sco[t+1,i,j]and not(np.isnan(stack_bool_sco[t,i,j])):
                matrix_sco[t,i,j]=sco_count+1
                sco_count=0
                
            if stack_bool_zeros[t,i,j]==stack_bool_zeros[t+1,i,j]:
                zeros_count+=1
            elif stack_bool_zeros[t,i,j]!=stack_bool_zeros[t+1,i,j]and not(np.isnan(stack_bool_zeros[t,i,j])):
                matrix_zeros[t,i,j]=zeros_count+1
                zeros_count=0
                
            if stack_bool[-1,i,j]==1:  
                matrix_dep[-1,i,j] = dep_count+1
            elif stack_bool[-1,i,j]==-1:
                matrix_sco[-1,i,j] = sco_count+1
            elif stack_bool[-1,i,j]==0:
                matrix_zeros[-1,i,j] = zeros_count+1

# print(stack_bool[:,yyy,xxx])
# print()
# print('matrix_dep ', matrix_dep[:,yyy,xxx])
# print('stack_bool_dep ', stack_bool_dep[:,yyy,xxx])
# print()
# print('matrix_sco ', matrix_sco[:,yyy,xxx])
# print('stack_bool_sco ', stack_bool_sco[:,yyy,xxx])
# print()
# print('matrix_zeros ', matrix_zeros[:,yyy,xxx])
# print('stack_bool_zeros ', stack_bool_zeros[:,yyy,xxx])

matrix_dep = np.where(matrix_dep==0, np.nan, matrix_dep)
matrix_sco = np.where(matrix_sco==0, np.nan, matrix_sco)
matrix_zeros = np.where(matrix_zeros==0, np.nan, matrix_zeros)

matrix_dep = matrix_dep.flatten()
matrix_sco = matrix_sco.flatten()
matrix_zeros = matrix_zeros.flatten()



# PLOTS
fig1, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=matrix_dep, binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Period length [run_tmie]',
       ylabel='Frequency',
       title='Number of consecutive depositional periods - '+run)
plt.show()

fig2, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=matrix_sco, binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Period length [run_tmie]',
       ylabel='Frequency',
       title='Number of consecutive erosional periods - '+run)
plt.show()

fig2, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=matrix_zeros, binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Period length [run_tmie]',
       ylabel='Frequency',
       title='Number of consecutive inactivity periods - '+run)
plt.show()


#%%
'''
This section operates on the stack structure to make possible to represent data on histograms
Furthermore this section create three different arrays as:
    act_tot_time_array: is the dataset of all the duration available on act_time_stack
    act_first_time_array: is the dataset of every first activation period for each pixel
    act_time_array: is the dataset of every activation period excluding the first one (the first one - as the last one,
                    to be clear - is just a lower limit of the "real" period because actually the true duration is unknow
                    since we were not measuring) 
'''
act_tot_time_array = act_time_stack.flatten() # Unroll active time array
act_tot_time_array = act_tot_time_array[act_tot_time_array !=0] # Trim zero values
act_tot_time_array = act_tot_time_array[np.logical_not(np.isnan(act_tot_time_array))] # Trim nan values

# act_first_time_array=act_first_time_array*mask
act_first_time_array = act_time_stack[0,:,:].flatten() # Unroll all active time
act_first_time_array = act_first_time_array[act_first_time_array !=0] # Trim zero values
act_first_time_array = act_first_time_array[np.logical_not(np.isnan(act_first_time_array))] # Trim nan values

# act_time_array=act_time_array*mask
act_time_array = act_time_stack[1:,:,:].flatten() # Unroll all active time exluding the first
act_time_array = act_time_array[act_time_array !=0] # Trim zero values
act_time_array = act_time_array[np.logical_not(np.isnan(act_time_array))] # Trim nan values

        
#%%
# FIRST SWITCH ACTIVATION TIME HISTOGRAM
# act_first_time_array
fig3, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=np.abs(act_first_time_array), binwidth=1, discrete=True, shrink=0.8) # , kde=True
ax.set(xlabel='Time between switches',
       ylabel='Count',
       title='First switch time - '+run)
# plt.ylim(0, 6000)
# set the ticks first
# ax.set_xticks(range(7))

ticks = []
for ti in range(0, int(np.nanmax(act_first_time_array))+1):
    ticks = np.append(ticks, str(ti))
    
# set the ticks first
ax.set_xticks(range(len(ticks)))

# set the labels
ax.set_xticklabels(ticks)
plt.savefig(os.path.join(home_dir, 'pixel_activity_plot', run + 'first_switch.pdf'))
plt.show()


print('Mean value of period:',  np.nanmean(np.abs(act_first_time_array)))
print('Median value of period:', np.nanmedian(np.abs(act_first_time_array)))

print('1:', np.nansum(np.abs(act_first_time_array)==1))
print('2:', np.nansum(np.abs(act_first_time_array)==2))
print('3:', np.nansum(np.abs(act_first_time_array)==3))
print('4:', np.nansum(np.abs(act_first_time_array)==4))
print('5:', np.nansum(np.abs(act_first_time_array)==5))
print('6:', np.nansum(np.abs(act_first_time_array)==6))
print('7:', np.nansum(np.abs(act_first_time_array)==7))
print('8:', np.nansum(np.abs(act_first_time_array)==8))
#%%    

# SWITCH ACTIVATION TIME EXCLUDING THE FIRST SWITCH
fig5, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=np.abs(act_time_array), binwidth=0.4, discrete=True, shrink=0.8)
# plt.ylim(0, 20000)
ax.set(xlabel='Time between switches',
       ylabel='Count',
       title='Switch time excluded the first one - '+run)
plt.show()
    
    
    
#%%
'''
This section could help you to have a double check on the period calcultaion
procedure.
Changing the indices the program will print the stack slice and the
corresponding activation period calculation
'''
print('Random choice array check:')
yy=random.randrange(dim_y)
xx=random.randrange(dim_x)
# Force check coordinates:
xx=93
yy=43
print(stack_bool[:,yy,xx])
print(act_time_stack[:,yy,xx])
print()



print('First switch time activation')
print('mean [min] = ', np.mean(np.abs(act_first_time_array)*dt))
print('STD [min] = ', np.std(act_first_time_array*dt))
print()
print('Total switch time activation')
print('mean [min] = ', np.mean(np.abs(act_tot_time_array)*dt))
print('STD [min] = ', np.std(act_tot_time_array*dt))
print()
print('Switch time activation excluding the first one')
print('mean [min] = ', np.mean(np.abs(act_time_array)*dt))
print('STD [min] = ', np.std(act_time_array*dt))
print()

#%%
'''
In this section I will try to eliminate the very short periods from the
act_time_stack to check if something change in the definition of the temporal scale.
'''
# first of all try to eliminate all the periods with duration equal to 1
# (or -1 if they are scou periods)

p_thr = 1 # periods length threshold

act_time_stack_1 = np.where(np.abs(act_time_stack)<=p_thr, 0, act_time_stack) # apply filter

act_time_array_1 = act_time_stack_1.flatten() # Unroll the period length array
act_time_array_1 = act_time_array_1[act_time_array_1 !=0] # Trim zero values
act_time_array_1 = act_time_array_1[np.logical_not(np.isnan(act_time_array_1))] # Trim nan values

print('Statistics over data without periods longer than ', p_thr, 'timestep')
print('mean [min] = ', np.mean(np.abs(act_time_array_1)*dt))
print('STD [min] = ', np.std(act_time_array_1*dt))
print()

# SMALL PERIOD ELIMINATION HISTOGRAM
fig5, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=np.abs(act_time_array_1), binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Time between switches',
       ylabel='Count',
       title='Activation period - '+run)
plt.show()

#%%
'''
In this section I will try to weigth the time period considering the magnitude
of scour and deposition. In particular, each period will be weighted by the
corresponding volume of scour or deposition.
'''
# Initializa arrays
periods_array = []
volumes_array = []
periods_stack = np.zeros(stack.shape) # This matrix will collect all the periods
periods_stack = np.where(periods_stack==0,np.nan,0) # Convert the matrix in a np.nan matrix
weighted_period_stack = np.zeros(stack.shape) # This stack will collect the activation period weighted by the ammount od scour or deposition volume correspondin at each period
weighted_period_stack = np.where(weighted_period_stack==0, np.nan,0) # Convert this matrx as a np.nan matrix
volumes_stack = np.zeros(stack.shape) # This stack will collect the scour and depositon volumes for each activation period
volumes_stack = np.where(volumes_stack==0,np.nan,0) # Convert this matrx as a np.nan matrix

w_act_time_stack = np.abs(act_time_stack) # convert the ative time stack as the absolute value (no differences between scour and deposition)
# I will create this matrix to chech if, where np.nan do not occour, the sum of all the period duration is equal to the total measuring period (for example 9)
# This matrix will help me to focus the nex steps of the analysis in the portion where a np.nan do not occour
sum_act_time_stack = np.sum(w_act_time_stack, axis=0)

indices = np.array(np.where(np.logical_not(np.isnan(sum_act_time_stack)))) # Indices where a np.nan do not occour
for i in range(0,len(indices[0,:])): # Loop all over the availabe indices
    y,x = indices[:,i]
    # x,y = 45,19
    stack_slice = np.abs(stack[:,y,x]) # Slice of the scour and deposition data
    period_slice = np.abs(act_time_stack[:,y,x]) # Slice of the correspondig period matrix
    period_slice_trim  = period_slice[period_slice!=0] # Trim zero values from the period array
    w_period_cumu = [] # Initialize cumulate period counting (it's just a process array)
    weight = [] # Initialize array where to stock a weight gor each period
    for i in range(0,len(period_slice_trim)): # This loop generate a comulative time array, it is useful to calculate the indices for further slicing operation
        w_period_cumu = np.append(w_period_cumu, np.sum(period_slice_trim[:i+1]))
    a = w_period_cumu - period_slice_trim # This array collects the lower inidices for slicing each period
    b = w_period_cumu # This array collects the upper indices for slicing each period
    
    for i in range(0,len(period_slice_trim)):
        array_sliced = stack_slice[int(a[i]):int(b[i])] # Perform slicing...
        # print(array_sliced)
        weight = np.append(weight, np.sum(array_sliced)) # ...and calculate the weight
    
    if len(weight) != len(period_slice_trim):
        print('STOP')
    
    # Create two arrays with all periods and volumes data
    periods_array = np.append(periods_array, period_slice_trim)
    volumes_array = np.append(volumes_array, weight)
    
        
    
    #TODO this operation should be checked!!
    period_weighted = period_slice_trim*weight/np.sum(weight) # Weighting operation
    
    # Fill the weight period and the volume matrix
    periods_stack[:len(period_slice_trim),y,x] = period_slice_trim
    weighted_period_stack[:len(period_weighted),y,x] = period_weighted
    volumes_stack[:len(weight),y,x] = weight

# The first reliable period in term oof length is the so called "second one"
# since the "first one" is just a lower limit.
# The "second period" is stored in the second place of the period stack:
period2_stack = periods_stack[1,:,:] # Consider the second period only
period2_array = period2_stack[period2_stack!=0] # Trim zero values
period2_array = period2_array[np.logical_not(np.isnan(period2_array))] # Trim np.nan values


# Where are placed the very short periods? They are quite a lot and could be
# interesting to investigate their position in relation of the envelop of scour and deposition
# areas

short_periods_matrix = np.where(periods_stack==1, 1, np.nan)
short_periods_matrix = np.nansum(short_periods_matrix, axis=0)
#######
# PLOT
#######
fig1, ax = plt.subplots(tight_layout=True)
short_periods_matrix = np.where(short_periods_matrix==0, np.nan, short_periods_matrix)
# shw1 = ax.imshow()
shw = ax.imshow(short_periods_matrix)
# # make bar
bar = plt.colorbar(shw) 
# # show plot with labels
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(run+'\n Spatial distribution of short periods (length 1)')
bar.set_label('Number of short periods')
plt.show()




# Weigthed periods
weighted_period2_stack = weighted_period_stack[1,:,:] # Consider only the second period (the firs complete one)

weighted_period_array = weighted_period2_stack[weighted_period2_stack!=0] # Trim zero values
weighted_period_array = weighted_period_array[np.logical_not(np.isnan(weighted_period_array))] # Trim np.nan values
# TODO This need to be checked:
# weighted_period_array = weighted_period_array[weighted_period_array>1] # Trim weighted array values lower than 1 

print()
print('First complete active weighted period')
print('mean [min] = ', np.mean(np.abs(weighted_period_array)*dt))
print('STD [min] = ', np.std(weighted_period_array*dt))
print()

###########
# PLOTS
###########

# SWITCH ACTIVATION TIME EXCLUDING THE FIRST SWITCH
fig6, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=np.abs(period2_array), binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Time between switches',
       ylabel='Count',
       title='First complete period - '+run)
plt.show()


# SWITCH ACTIVATION WEIGHTED TIME
fig7, ax = plt.subplots(tight_layout=True)
ax = sns.histplot(data=np.abs(weighted_period_array), binwidth=0.4, discrete=True, shrink=0.8)
ax.set(xlabel='Weighted time between switches',
       ylabel='Count',
       title='First complete weighted period - '+run)
plt.show()

#%%
'''
Do the most tiny scour and deposition volumes are referred to the smallest periods?
In other words, there is as correlation between the volumes of scou and depositon and the time period?
Let's try to find thi correlation: to do so, I decide to plot the time period and the associated ammount of scour or deposition volume.
'''

# # PERIODS
# periods = np.abs(periods_stack) # Evaluate the absolute value
# periods = periods.flatten() # Make the stack flat
# periods = periods[periods!=0] # Trim zero values
# periods = periods[np.logical_not(np.isnan(periods))] # Trim nan values

# # VOLUMES OF SCOUR OR DEPOSITION
# volumes = np.abs(volumes_stack) # Evaluate the absolute value
# volumes = volumes.flatten() # Make the stack flat
# volumes = volumes[volumes!=0] # Trim zero values
# volumes = volumes[np.logical_not(np.isnan(volumes))] # Trim nan values

print(len(periods_array), len(volumes_array))


# PLOT
fig6, ax = plt.subplots(tight_layout=True)
plt.scatter(periods_array, volumes_array, marker='+')
ax.set(xlabel='Periods',
       ylabel='Volumes',
       title='Correlation between period lenght and volumes')
plt.show()



# # Boxplot:
# fig, ax = plt.subplots(dpi=80, figsize=(10,6))
# fig.suptitle('Correlation between period lenght and volumes - '+run, fontsize = 18)  
# for m in range(int(periods_array.min()),int(periods_array.max())+1):
#     period_volume_matrix = np.vstack((periods_array, volumes_array))
#     for i in range(0,len(periods_array)):
#         if period_volume_matrix[0,i] !=m or np.isnan(period_volume_matrix[0,i]).any():
#             period_volume_matrix[:,i]=[np.nan,np.nan]
#     period_volume_matrix=period_volume_matrix[np.logical_not(np.isnan(period_volume_matrix))]  # Trim np.nan values (this procedure flatten the matrix)
#     period_volume_matrix=period_volume_matrix[period_volume_matrix!=0] # Trim zero values
#     period_volume_matrix = np.vstack((period_volume_matrix[:int(len(period_volume_matrix)*0.5)], period_volume_matrix[int(len(period_volume_matrix)*0.5):])) # Here I rebuilt the 2D Matrix
    
#     # bplot=ax.boxplot(period_volume_matrix[1,:], positions=[m], widths=0.5) # Data were filtered by np.nan values
#     bplot=ax.boxplot(np.log(period_volume_matrix[1,:]), positions=[m], widths=0.5) # Data were filtered by np.nan values
# ax.yaxis.grid(True)
# # ax.set_yscale('log')
# ax.set_xlabel('Time periods', fontsize=12)
# ax.set_ylabel('ln Time period volume', fontsize=12)
# plt.xticks(np.arange(int(periods_array.min()),int(periods_array.max())+1, 1))
# # plt.savefig(os.path.join(plot_dir, 'morphWact_boxplot.png'), dpi=200)
# plt.show()
#%%
# Inviluppo: numero di pixel che sono stati attivi almeno una volta
sum_matrix = np.sum(np.abs(stack_bool), axis=0)
sum_matrix = np.sum(np.where(sum_matrix!=0,1,sum_matrix))


end = time.time()
print()
print('Execution time: ', (end-start), 's')