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




# SINGLE RUN NAME
run = 'q07_1'
# Step between surveys
DoD_delta = 1

# setup working directory and DEM's name
home_dir = os.getcwd()
# Source DoDs folder
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack')

stack_name = 'DoD_stack_bool' + str(DoD_delta) + '_' + run + '.npy' # Define stack name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack = np.load(stack_path) # Load DoDs stack

dim_t, dim_y, dim_x = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension

matrix0 = stack[0,:,:]
for t in range(1,dim_t):
    matrix = np.multiply(matrix0, stack[t,:,:])
    matrix0 = matrix
    
pixel_tot = dim_x*dim_y - np.sum(np.isnan(matrix0)) # Pixel domain without considering NaN value



# Pixel attivi sia all'inizio che alla fine
a = np.nansum(abs(np.multiply(stack[0,:,:],stack[dim_t-1,:,:])))/pixel_tot

# Pixel attivi all'inizio ma non alla fine
b = (np.nansum(abs(stack[0,:,:])) - np.nansum(abs(np.multiply(stack[0,:,:],stack[dim_t-1,:,:]))))/pixel_tot

# Pixel attivi alla fine ma non all'inizio
c = (np.nansum(abs(stack[dim_t -1,:,:])) - np.nansum(abs(np.multiply(stack[0,:,:],stack[dim_t-1,:,:]))))/pixel_tot

# Pixel attivi nè all'inizio nè alla fine
d = dim_x*dim_y/pixel_tot - (a+b+c)


