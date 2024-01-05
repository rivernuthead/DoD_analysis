#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:38:09 2023

@author: erri

This script compute the DoD envelope.

DoD input stack structure:
    

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
        

        



OUTPUT:

"""

import os
import numpy as np

set_names = ['q07_1','q10_2', 'q10_3', 'q10_4', 'q15_2', 'q15_3', 'q20_2']
# set_names = ['q20_2']

for set_name in set_names:
    print(set_name, ' is running...')
    # IMPORT DoD STACK AND DoD BOOL STACK
    home_dir = os.getcwd() # Home directory
    DoDs_folder = os.path.join(home_dir, 'output', 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    stack_act = np.abs(stack_bool)
    stack_sco = stack*(stack<0)
    stack_dep = stack*(stack>0)
    stack_sco_bool = stack_bool*(stack<0)
    stack_dep_bool = stack_bool*(stack>0)
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # COMPUTE THE DOMAIN WIDTH ARRAY
    # This array contains the domain width of every cross section
    domain = abs(np.copy(stack_bool))
    domain = np.where(domain>=0,1,np.nan)
    domain_width_stack = np.nansum(domain,axis=1)
    
    '''
    COMPUTE THE NET VOLUME    
    '''
    report_net_volume = np.nansum(stack, axis=2)
    report_net_volume = np.nansum(report_net_volume, axis=1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_net_volume.txt'), report_net_volume, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE SCOUR VOLUME
    '''
    stack_scour_volume = np.where(stack<0, stack, 0)
    report_scour_volume = np.nansum(stack_scour_volume, axis=2)
    report_scour_volume = np.nansum(report_scour_volume, axis=1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_scour_volume.txt'), report_scour_volume, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE FILL VOLUME
    '''
    stack_fill_volume = np.where(stack>0, stack, 0)
    report_fill_volume = np.nansum(stack_fill_volume, axis=2)
    report_fill_volume = np.nansum(report_fill_volume, axis=1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_fill_volume.txt'), report_fill_volume, delimiter=',', fmt = "%.4f")
    
    
    '''
    COMPUTE THE MOIRPHOLOGICAL ACTIVE WIDTH
    '''
    stack_act = abs(stack_bool)
    report_MAW = np.nansum(stack_act, axis=2)
    report_MAW = np.nansum(report_MAW, axis=1)
    report_MAW = report_MAW/stack.shape[2]/120
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_MAW.txt'), report_MAW, delimiter=',', fmt = "%.4f")
    
    
    '''
    COMPUTE THE SCOUR MORPHOLOGICAL ACTIVE WIDTH
    '''
    stack_sco_act = abs(stack_sco_bool)
    report_sco_MAW = np.nansum(stack_sco_act, axis=2)
    report_sco_MAW = np.nansum(report_sco_MAW, axis=1)
    report_sco_MAW = report_sco_MAW/stack.shape[2]/120
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_sco_MAW.txt'), report_sco_MAW, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE FILL MORPHOLOGICAL ACTIVE WIDTH
    '''
    stack_dep_act = abs(stack_dep_bool)
    report_dep_MAW = np.nansum(stack_dep_act, axis=2)
    report_dep_MAW = np.nansum(report_dep_MAW, axis=1)
    report_dep_MAW = report_dep_MAW/stack.shape[2]/120
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_fill_MAW.txt'), report_dep_MAW, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE AVERAGE ACTIVE THICKNESS
    '''
    stack_thickness = np.abs(np.copy(stack))
    stack_thickness = np.where(stack_thickness==0, np.nan, stack_thickness)
    report_MAT = np.nanmean(stack_thickness, axis = 2)
    report_MAT = np.nanmean(report_MAT, axis = 1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_MAT.txt'), report_MAT, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE SCOUR AVERAGE ACTIVE THICKNESS
    '''
    stack_sco_thickness = np.abs(np.copy(stack_sco))
    stack_sco_thickness = np.where(stack_sco_thickness==0, np.nan, stack_sco_thickness)
    report_sco_MAT = np.nanmean(stack_sco_thickness, axis = 2)
    report_sco_MAT = np.nanmean(report_sco_MAT, axis = 1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_sco_MAT.txt'), report_sco_MAT, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE FILL AVERAGE ACTIVE THICKNESS
    '''
    stack_dep_thickness = np.abs(np.copy(stack_dep))
    stack_dep_thickness = np.where(stack_dep_thickness==0, np.nan, stack_dep_thickness)
    report_dep_MAT = np.nanmean(stack_dep_thickness, axis = 2)
    report_dep_MAT = np.nanmean(report_dep_MAT, axis = 1)
    np.savetxt(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_fill_MAT.txt'), report_dep_MAT, delimiter=',', fmt = "%.4f")
    
    '''
    COMPUTE THE NEW ACTIVATED AREA FOR EACH DoD
    '''
    # 1=activated, -1=deactivated, 0=no_changes
    stack_diff = stack_act[1:,:,:,:] - stack_act[:-1,:,:,:]
    np.save(os.path.join(home_dir, 'output','report_'+set_name, set_name + '_diff_stack.npy'),stack_diff)