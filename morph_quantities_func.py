#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:45:38 2022

@author: erri
"""
import numpy as np
import os
run = 'q07_1'
DoD_name = 'DoD_s1-s0_filt_nozero_rst.txt'
home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')
DoD_path = os.path.join(DoDs_dir, 'DoD_' + run, DoD_name)
DoD = np.loadtxt(DoD_path, delimiter='\t')

array = np.where(DoD==-999, np.nan, DoD)

# # Define nature array as -1=sco, 0=no_changes, and 1=dep
# nature_array = np.where(array>0, 1, array)
# nature_array = np.where(nature_array<0, -1, nature_array)


def morph_quantities(array):
    import numpy as np
    '''
    This function ...
    
    Input:
        array: 2D numpy array
            2D array with np.nans insted of -999.

    '''
    
    # Define total volume matrix, Deposition matrix and Scour matrix
    vol_array = np.where(np.isnan(array), 0, array) # Total volume matrix
    dep_array = (vol_array>0)*vol_array # DoD of only deposition data
    sco_array = (vol_array<0)*vol_array # DoD of only scour data
    
    tot_vol = np.sum(vol_array) # Total net volume [L³]
    sum_vol = np.sum(np.abs(vol_array)) # Sum of scour and deposition volume as V [l³]
    dep_vol = np.sum(dep_array) # Deposition volume  [L³]
    sco_vol = np.sum(sco_array) # Scour volume [L³]
    
    # Define nature array as -1=sco, 0=no_changes, and 1=dep
    nature_array = np.where(array>0, 1, array)
    nature_array = np.where(nature_array<0, -1, nature_array)
    
    # Define activity array:
    tot_act_array = nature_array*(nature_array!=0) # Where active then 1
    dep_act_array = nature_array*(nature_array>0) # Where scour then 1
    sco_act_array = nature_array*(nature_array<0) # Where scour then 1
    
    # Calculate morphological quantities
    morph_act_area = np.count_nonzero(tot_act_array) # Active area both in terms of scour and deposition in number of cells [-]
    morph_act_area_dep = np.count_nonzero(dep_act_array) # Active deposition area in number of cells [-]
    morph_act_area_sco = np.count_nonzero(sco_act_array) # Active scour area in number of cells [-]
    
    act_width_array = np.array([np.nansum(tot_act_array, axis=0)]) # Array of the crosswise morphological total active width in number of cells
    act_width_array_dep = np.array([np.nansum(dep_act_array, axis=0)]) # Array of the crosswise morphological deposition active width in number of cells
    act_width_array_sco = np.array([np.nansum(sco_act_array, axis=0)]) # Array of the crosswise morphological scour active width in number of cells
    
    act_width_mean = np.nanmean(act_width_array) # Total mean active width in number of cells (could be a real number)
    act_width_mean_dep = np.nanmean(act_width_array_dep) # Deposition mean active width in number of cells (could be a real number)
    act_width_mean_sco = np.nanmean(act_width_array_sco) # Scour mean active width in number of cells (could be a real number)
    
    # Calculate active thickness for total volumes. deposition volumes and scour volumes
    act_thickness_dep = np.nanmean(np.abs(dep_array)) # Deposition active thickness (abs(V_sco) + V_dep)/act_area [mm]
    act_thickness_sco = np.nanmean(np.abs(sco_array)) # Scour active thickness (abs(V_sco) + V_dep)/act_area [mm]
    
    return tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco

tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco = morph_quantities(array)