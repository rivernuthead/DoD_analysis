#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:38:09 2023

@author: erri

This script compute the DoD envelope.

INPUT:
    the script takes in input a stack where all the DoDs are stored in a structure as shown below:
        
    DoD input stack structure:
        
        DoD_stack[time,y,x,delta]
        DoD_stack_bool[time,y,x,delta]
        
         - - - 0 - - - - 1 - - - - 2 - - - - 3 - - - - 4 - - - - 5 - - - - 6 - - - - 7 - - - - 8 - -  >    delta
      0  |  DoD 1-0   DoD 2-0   DoD 3-0   DoD 4-0   DoD 5-0   DoD 6-0   DoD 7-0   DoD 8-0   DoD 9-0
      1  |  DoD 2-1   DoD 3-1   DoD 4-1   DoD 5-1   DoD 6-1   DoD 7-1   DoD 8-1   DoD 9-1
      2  |  DoD 3-2   DoD 4-2   DoD 5-2   DoD 6-2   DoD 7-2   DoD 8-2   DoD 9-2
      3  |  DoD 4-3   DoD 5-3   DoD 6-3   DoD 7-3   DoD 8-3   DoD 9-3
      4  |  DoD 5-4   DoD 6-4   DoD 7-4   DoD 8-4   DoD 9-4
      5  |  DoD 6-5   DoD 7-5   DoD 8-5   DoD 9-5
      6  |  DoD 7-6   DoD 8-6   DoD 9-6
      7  |  DoD 8-7   DoD 9-7
      8  |  DoD 9-8
         |
         v
        
         time
    
"""

import os
import numpy as np

set_names = ['q07_1','q10_2', 'q10_3', 'q10_4', 'q15_2', 'q15_3', 'q20_2']
# set_names = ['q20_2']

for set_name in set_names:
    # IMPORT DoD STACK AND DoD BOOL STACK
    home_dir = os.getcwd() # Home directory
    DoDs_folder = os.path.join(home_dir, 'output', 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # Create activity map from the stack (1=active, 0=inactive)
    act_stack = np.abs(stack_bool)
    
    # Compute the envelope at the finest timestep
    
    envBAW_map = np.asarray(act_stack[0,:,:,0], dtype='float64')
    envBAW_array = np.array(np.nansum(envBAW_map)/dim_x/120)
    for t in range(1,dim_t):
        envBAW_map =  envBAW_map + act_stack[t,:,:,0]
        envBAW_map = (envBAW_map>0)*1
        envBAW = np.nansum(envBAW_map)/dim_x/120
        envBAW_array = np.append(envBAW_array,envBAW)
    
    print(set_name)
    print(envBAW_array)
    print()
    
    
    
    
    
    
    
    
    
    
#     # COMPUTE THE DOMAIN WIDTH ARRAY
#     # This array contains the domain width of every cross section
#     domain = abs(stack_bool[:,:,:,:])
#     domain = np.where(domain==0,1,domain)
#     domain_width_stack = np.nansum(domain,axis=1)
    
#     # COMPUTE THE ACTIVITY STACK AS ACTIVE (1) AND INACTIVE (-1)
#     stack_act = abs(stack_bool) # Create a stack that shows activity only (1,0)
    
    
#     # COMPUTE THE ENVELOPE ARRAY AT THE FINEST TIMESTEP
#     stack_act_1 = stack_act[:,:,:,0]
#     envelope_1_array = []
#     for t in range(1, stack_act_1.shape[0]+1):
#         envelope = np.nansum(stack_act_1[:t,:,:], axis=0)
#         envelope_act = np.where(envelope>0,1,0)
#         envMAW = np.nansum(envelope_act)/(stack_act_1.shape[2]*120)
#         envelope_1_array = np.append(envelope_1_array, envMAW)
        
#     # SAVE TXT REPORT
#     np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_envMAW.txt'), np.round(envelope_1_array, decimals=3), delimiter=',')
    
#     # COMPUTE THE STACK ENVELOPE FOLLOWING THE DATASTRUCTURE AS BELOW
#     stack_act = abs(stack_bool) # Create a stack that shows activity only (1,0)
    
#     stack_env = stack_act[:-1,:,:,:-1]+stack_act[1:,:,:,:-1]
    
    
#     # SET TO ZERO THE ENVELOPE COMPUTED OUT OF THE DOMAIN
    
#     # Set the lower half of the diagonal to zero
#     for i in range(dim_t-1):
#         for j in range(dim_delta-1):
#             if i+j+1>dim_t:
#                 stack_env[i,:,:,j] = np.nan
    
    
    
#     # Create the stack as active and inactive areas
#     stack_env_bool = np.where(stack_env>0,1,stack_env)
    
#     # SAVE ENVELOPES ARCHIVE
#     np.save(os.path.join(home_dir, 'output','DoDs_envelope', set_name + '_stack_env.npy'), stack_env)
#     np.save(os.path.join(home_dir, 'output','DoDs_envelope', set_name + '_stack_env_bool.npy'), stack_env_bool)
    
    
    
#     # Testing:
#     # DoD1 = stack_act[0,:,:,0]
#     # DoD2 = stack_act[1,:,:,0]
    
#     # envDOD12 = DoD1+DoD2
    
#     # check = stack_env[0,:,:,0]
    
# #%%
#     '''
#     This section computes:
#         1. The MAW from the envelopes
#         2. The MAW using an increasing number of DoDs
#     '''
    
#     # 1.
#     stack_envMAW = np.nansum(stack_env_bool, axis=1)
#     stack_envMAW = stack_envMAW/domain_width_stack[:-1,:,:-1]
    
#     report_envMAW_mean = np.nanmean(stack_envMAW, axis=1)
#     report_envMAW_std = np.std(stack_envMAW, axis=1)
    
    
#     # Save txt report
#     np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_envMAW_mean.txt'), report_envMAW_mean, delimiter=',')
#     np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_envMAW_std.txt'), report_envMAW_std, delimiter=',')

    
#     # 2.
    
