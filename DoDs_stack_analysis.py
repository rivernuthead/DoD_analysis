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

#%%
start = time.time() # Set initial time


# SINGLE RUN NAME
run = 'q07_1'
print('###############\n' + '#    ' + run + '    #' + '\n###############')
# Step between surveys
DoD_delta = 1

# setup working directory and DEM's name
home_dir = os.getcwd()
# Source DoDs folder
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack')

stack_name = 'DoD_stack' + str(DoD_delta) + '_' + run + '.npy' # Define stack name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack = np.load(stack_path) # Load DoDs stack

# Create boolean stack # 1=dep, 0=no changes, -1=sco
stack_bool = np.where(stack>0, 1, stack)
stack_bool = np.where(stack<0, -1, stack_bool)

dim_t, dim_y, dim_x = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension


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

# Deposition pixel On matrix (1,1,0)
dep_px_On = (P[:,:,:,0]==1)*(P[:,:,:,1]==1)*(P[:,:,:,2]==0)
dep_px_On_count = np.sum(dep_px_On, axis=0)

# fig, ax = plt.subplots()
# ax.hist(dep_px_On_count.flatten())

# Deposition pixel Off matrix (1,-1,0)
dep_px_Off = (P[:,:,:,0]==1)*(P[:,:,:,1]==-1)*(P[:,:,:,2]==0)
dep_px_Off_count = np.sum(dep_px_Off, axis=0)

# Scour pixel On matrix (-1,-1,0)
sco_px_On = (P[:,:,:,0]==-1)*(P[:,:,:,1]==-1)*(P[:,:,:,2]==0)
sco_px_On_count = np.sum(sco_px_On, axis=0)

# Scour pixel Off matrix (-1,1,0)
sco_px_Off = (P[:,:,:,0]==-1)*(P[:,:,:,1]==1)*(P[:,:,:,2]==0)
sco_px_Off_count = np.sum(sco_px_Off, axis=0)

# Changes from Scour to Deposition (0,2,-1)
sco2dep_px = (P[:,:,:,0]==0)*(P[:,:,:,1]==2)*(P[:,:,:,2]==-1)
sco2dep_px_count = np.sum(sco2dep_px, axis=0)

# Changes from Deposition to Scour (0,-2,-1)
dep2sco_px = (P[:,:,:,0]==0)*(P[:,:,:,-2]==1)*(P[:,:,:,2]==-1)
dep2sco_px_count = np.sum(dep2sco_px, axis=0)

# Permanence of Deposition (2,0,1)
dep_px = (P[:,:,:,0]==2)*(P[:,:,:,2]==0)*(P[:,:,:,2]==1)
dep_px_count = np.sum(dep_px, axis=0)

# Permanence of Scour (-2,0,1)
sco_px = (P[:,:,:,0]==-2)*(P[:,:,:,2]==0)*(P[:,:,:,2]==1)
sco_px_count = np.sum(sco_px, axis=0)

# Permanece of NoChanges (0,0,0)
still_px = (P[:,:,:,0]==0)*(P[:,:,:,2]==0)*(P[:,:,:,2]==0)
still_px_count = np.sum(still_px, axis=0)



# Activation time
# Time needed for the first activation




end = time.time()
print()
print('Execution time: ', (end-start), 's')