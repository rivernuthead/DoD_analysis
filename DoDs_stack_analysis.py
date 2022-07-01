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
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat


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


# Two by two sum
# where abs(sum2x2)=1, new pixel activation
sum2x2 = stack_bool[:-1,:,:]+stack_bool[1:,:,:]
new_act_stack = np.where(abs(sum2x2)==1, 1, 0)
new_act_map = np.sum(new_act_stack, axis=0)

# Two by two multilication
# where mult2x2=-1, change in cell nature (scour->deposition or deposition->scour)
mult2x2 = stack_bool[:-1,:,:]*stack_bool[1:,:,:]
nature_change_stack = np.where(mult2x2==-1, 1, 0)
nature_change_map = np.sum(nature_change_stack, axis=0)








# Activation time
# Time needed for the first activation
