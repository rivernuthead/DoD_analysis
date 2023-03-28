#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

Pixel age analysis over stack DoDs
"""

import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns

#%%
start = time.time() # Set initial time

# Script mode
plot_mode = 1

# SINGLE RUN NAME
run = 'q07_1'
print('###############\n' + '#    ' + run + '    #' + '\n###############')
# Step between surveys
DoD_delta = 1

# setup working directory and DEM's name
home_dir = os.getcwd()
# Source DoDs folder
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack')

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

stack_name = 'DoD_stack' + str(DoD_delta) + '_' + run + '.npy' # Define stack name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack = np.load(stack_path) # Load DoDs stack

# Initialize stack
act_time_stack = np.zeros(stack.shape) # activation time stack contains the time between switches. The first layer of this stack contains the first sctivation time that is a lower limit in time because we ignore how long the pixel has keept the same nature in the past.
switch_matrix = np.zeros(stack.shape[1:]) # This is the 2D matrix that collect the number of switch over time


# Create boolean stack # 1=dep, 0=no changes, -1=sco
stack_bool = np.where(stack>0, 1, stack)
stack_bool = np.where(stack<0, -1, stack_bool)

dim_t, dim_y, dim_x = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension

# Mask
mask = np.sum(abs(stack_bool), axis=0)
mask = np.where(np.isnan(mask), mask,1)


for t in range(0,dim_t):
    if t == 0:
        matrix0 = stack_bool[0,:,:]
    else:
        pass
    matrix = np.multiply(matrix0, stack_bool[t,:,:])
    matrix0 = matrix
    
pixel_tot = dim_x*dim_y - np.sum(np.isnan(matrix0)) # Pixel domain without considering NaN value



# Pixel attivi sia all'inizio che alla fine
a = np.nansum(abs(np.multiply(stack_bool[0,:,:],stack_bool[dim_t-1,:,:])))/pixel_tot

# Pixel attivi all'inizio ma non alla fine
b = (np.nansum(abs(stack_bool[0,:,:])) - np.nansum(abs(np.multiply(stack_bool[0,:,:],stack_bool[dim_t-1,:,:]))))/pixel_tot

# Pixel attivi alla fine ma non all'inizio
c = (np.nansum(abs(stack_bool[dim_t -1,:,:])) - np.nansum(abs(np.multiply(stack_bool[0,:,:],stack_bool[dim_t-1,:,:]))))/pixel_tot

# Pixel attivi nè all'inizio nè alla fine
d = dim_x*dim_y/pixel_tot - (a+b+c)

# Active pixel
e = np.nansum(abs(stack_bool), axis=0)

# Active area for each DoD
act_A = []
for t in range(0,dim_t):
    DoD_act_A = np.nansum(abs(stack_bool[t,:,:]))
    act_A = np.append(act_A, DoD_act_A) # Active area array for each DoD in the stack

# Number of activated pixel from a DoD and the consecutive one
activated_pixels = []
for t in range(0,dim_t-1):
    activated_pixel_count = np.nansum(abs(stack_bool[t+1,:,:])) - np.nansum(abs(np.multiply(stack_bool[t,:,:],stack_bool[t+1,:,:])))
    activated_pixels = np.append(activated_pixels, activated_pixel_count)

#%%
# Create the data structure to perform sum, diff and mult two by two along the time axis

# Create SUM, DIFFERENCE and MULTIPLICATION matrix:
# matrix = matrix[t,y,x]
sum_matrix = stack_bool[1:,:,:] + stack_bool[:-1,:,:] # SUM matrix
dif_matrix = stack_bool[1:,:,:] - stack_bool[:-1,:,:] # DIFFERENCE matrix
mul_matrix = stack_bool[1:,:,:]*stack_bool[:-1,:,:] # MULTIPLICATION matrix

# Create P matrix
# matrix = matrix[t,y,x,i]
# i=0 sum_matrix, i=1 dif_matrix, i=2 mul_matrix
dim = np.append(sum_matrix.shape, 3) # Matrix P dimension
P = np.zeros(dim) # Initialize P
P[:,:,:,0] = sum_matrix 
P[:,:,:,1] = dif_matrix
P[:,:,:,2] = mul_matrix

#%%

# Deposition pixel switching On matrix (1,1,0)
dep_px_On = (P[:,:,:,0]==1)*(P[:,:,:,1]==1)*(P[:,:,:,2]==0)
dep_px_On_count = np.sum(dep_px_On, axis=0)

# fig, ax = plt.subplots()
# ax.hist(dep_px_On_count.flatten())

# Deposition pixel switching Off matrix (1,-1,0)
dep_px_Off = (P[:,:,:,0]==1)*(P[:,:,:,1]==-1)*(P[:,:,:,2]==0)
dep_px_Off_count = np.sum(dep_px_Off, axis=0)

# Scour pixel switching On matrix (-1,-1,0)
sco_px_On = (P[:,:,:,0]==-1)*(P[:,:,:,1]==-1)*(P[:,:,:,2]==0)
sco_px_On_count = np.sum(sco_px_On, axis=0)

# Scour pixel switching Off matrix (-1,1,0)
sco_px_Off = (P[:,:,:,0]==-1)*(P[:,:,:,1]==1)*(P[:,:,:,2]==0)
sco_px_Off_count = np.sum(sco_px_Off, axis=0)

# Changes from Scour to Deposition (0,2,-1)
sco2dep_px = (P[:,:,:,0]==0)*(P[:,:,:,1]==2)*(P[:,:,:,2]==-1)
sco2dep_px_count = np.sum(sco2dep_px, axis=0)

# Changes from Deposition to Scour (0,-2,-1)
dep2sco_px = (P[:,:,:,0]==0)*(P[:,:,:,1]==-2)*(P[:,:,:,2]==-1)
dep2sco_px_count = np.sum(dep2sco_px, axis=0)

# Permanence of Deposition (2,0,1)
dep_px = (P[:,:,:,0]==2)*(P[:,:,:,1]==0)*(P[:,:,:,2]==1)
dep_px_count = np.sum(dep_px, axis=0)

# Permanence of Scour (-2,0,1)
sco_px = (P[:,:,:,0]==-2)*(P[:,:,:,1]==0)*(P[:,:,:,2]==1)
sco_px_count = np.sum(sco_px, axis=0)

# Permanece of NoChanges (0,0,0)
still_px = (P[:,:,:,0]==0)*(P[:,:,:,1]==0)*(P[:,:,:,2]==0)
still_px_count = np.sum(still_px, axis=0)


#%%
# ACTIVATION TIME
# This loop works all over the x and y spatial coordinates.
for x in range(0,dim_x):
    for y in range(0,dim_y):
        slice_array = stack_bool[:,y,x] # Slice the stack in a single pixel array where data is collected over time
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
            if not slice_array.any(): # Check if the sliced array is full of zeros
                time_array = np.array([np.nan]) # Fill the array with np.nan to keep them transparent
            else:
                n_zero=np.array(np.where(slice_array!=0))[0,0] # Number of zero values before the first non-zero value
                
                slice_array=slice_array[n_zero:] # Trim the zero values before the first non-zero value.
                
        time_array = []
        if slice_array[0]!=0: # If the first entry of the sliced array is non-zero
            count=0 # Initialize the count variable. This variable will count the number of activation instants
            target_sign = np.sign(slice_array[0]) # This variable collects the sign of the first element of each same-nature period
            for i in range(0,len(slice_array)-1): # Loop over the sliced array
                
                a1, a2 = slice_array[i], slice_array[i+1] # a1 and a2 are the two adjacent element in the sliced array
                
                if np.sign(a1)==np.sign(a2): # If two consecutive elements have the same naure
                    count += 1 # If two consecutive elements have the same nature the count increases
                elif np.sign(a1)*np.sign(a2)==0 and (np.sign(a2)==target_sign or np.sign(a1)==target_sign):
                    count += 1 # The count increases also if one or both elements are zero but the non-zero value has the target sign
                elif np.sign(a1)!=np.sign(a2) and (np.sign(a2)!=target_sign or np.sign(a1)!=target_sign): # The count stops when a switch occours or when one of the two elements shows a sign different from the target sign
                    time_array = np.append(time_array, count*target_sign)  # This operation apped to time_array the count value with his sign. This could be useful to keep trace of the nature of the period.
                    target_sign=-1*target_sign # Update the target sign
                    count=0 # Update the count variable that will starts again from zero
                    pass
                
        time_array = time_array+1*np.sign(time_array) # This operation update the right time due to zero index rapresentation
        time_array = np.append(time_array, (len(slice_array)-np.sum(np.abs(time_array)))*target_sign) # By now the last period is not calculated (actually because, as the first one, it is only a lower boundary of time because it doesn't appear within two switches) so this operation appeds this value manually
        time_array[0] = time_array[0] + np.sign(time_array[0])*n_zero # Ths operation append, if present, the number of zeroes before the first non-zero value calculated on the very first sliced array (n_zero variable)
        
        
        act_time_stack[:len(time_array),y,x]=time_array # his operation fills the stack with time_array

        
        if len(time_array)==0 or len(time_array)==1: # If sliced array does not contain any switch (so if the length of the time array is 0 in the case we do not consider the last period - see above - or if the length is 1 so only one period is considered - the last one - this operation fills the output matrix with np.nan)
            switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                act_time_stack[t,y,x]=np.nan
        
        elif len(time_array)==2: # If only one switch occours, two periods were identified. This two periods are not completely reliable since they were not observed between two switches
            switch_matrix[y,x] = np.nan # Fill switch matrix with np.nan
            for t in range(0,act_time_stack.shape[0]): # Fill activation time stack with np.nan
                act_time_stack[t,y,x]=np.nan
            
        else: # If at least one (or two) switches occours
            # Fill switch matrix
            switch_matrix[y,x] = len(time_array) # To provide the number of switch
            
            # Fill activation time stack
            act_time_stack[:len(time_array),y,x]=time_array # To provide the time between each detected switch
 
#%%


# act_time_stack_diff = act_time_stack[1:,:,:]-act_time_stack[:-1,:,:] # Calculate interval between switch
# act_time_stack_diff = np.where(act_time_stack_diff>0, act_time_stack_diff, 0) # Keep only positive difference

# act_time_stack = np.vstack((np.resize(act_time_stack[0,:,:], (1,dim_y,dim_x)),act_time_stack_diff))


# # Apply mask:
# for t in range(0,act_time_stack.shape[0]):
#     act_time_stack[t,:,:] = act_time_stack[t,:,:]*mask

# Unroll arrays
# act_tot_time_array=act_tot_time_array
act_tot_time_array = act_time_stack.flatten() # Unroll all active time
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
###########
# PLOTS
###########
if plot_mode ==1:
    for i in range(0,5):
        # ACTIVE PIXEL AT LEAST ONCE
        fig1, ax = plt.subplots(tight_layout=True)
        # e=e*mask
        e = np.where(e==0,np.nan,e) # Make 0 value as np.nan (100% transparency)
        e1 = np.where(e<=i,np.nan,e) # Mask pixel active less than i time
        shw = ax.imshow(e1)
        # # make bar
        # bar = plt.colorbar(shw) 
        # # show plot with labels
        # plt.xlabel('X coordinate')
        # plt.ylabel('Y coordinate')
        # plt.title(run)
        # bar.set_label('Number of active pixel')
        plt.show()
    
    # NUMBER OF SWITCH
    fig2, ax = plt.subplots(tight_layout=True)
    # e=e*mask
    shw = ax.imshow(np.where(np.isnan(switch_matrix),0,switch_matrix))
    # make bar
    bar = plt.colorbar(shw)
    # show plot with labels
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(run)
    bar.set_label('Number of pixel switch')
    plt.show()

    # FIRST SWITCH ACTIVATION TIME HISTOGRAM
    # act_first_time_array
    fig3, ax = plt.subplots(tight_layout=True)
    ax = sns.histplot(data=act_first_time_array, binwidth=0.4, discrete=True, shrink=0.8)
    ax.set(xlabel='Time between switches',
           ylabel='Count',
           title='First switch time - '+run)
    plt.show()
    
    # SWITCH ACTIVATION TIME EXCLUDING THE FIRST SWITCH
    fig5, ax = plt.subplots(tight_layout=True)
    ax = sns.histplot(data=act_time_array, binwidth=0.4, discrete=True, shrink=0.8)
    ax.set(xlabel='Time between switches',
           ylabel='Count',
           title='Switch time excluded the first one - '+run)
    plt.show()
    
    # SWITCH ACTIVATION TIME HISTOGRAM
    fig4, ax = plt.subplots(tight_layout=True)
    ax = sns.histplot(data=act_tot_time_array, binwidth=0.4, discrete=True, shrink=0.8)
    ax.set(xlabel='Time between switches',
           ylabel='Count',
           title='Switch time - '+run)
    plt.show()
    
    
    



#%%
yy=36
xx=6
print(stack_bool[:,yy,xx])
print(act_time_stack[:,yy,xx])
print()
print('First switch time activation')
print('mean [min] = ', np.mean(act_first_time_array*dt))
print('STD [min] = ', np.std(act_first_time_array*dt))
print()
print('Total switch time activation')
print('mean [min] = ', np.mean(act_tot_time_array*dt))
print('STD [min] = ', np.std(act_tot_time_array*dt))
print()
print('Switch time activation excluding the first one')
print('mean [min] = ', np.mean(act_time_array*dt))
print('STD [min] = ', np.std(act_time_array*dt))

#%%
end = time.time()
print()
print('Execution time: ', (end-start), 's')