#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:45:38 2022

@author: erri
"""

import numpy as np
import os

###########
# TEST PART
###########
run = 'q07_1' # Run name
DoD_name = 'DoD_s1-s0_filt_nozero_rst.txt' # DoD name
# Set DoD path:
home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')
DoD_path = os.path.join(DoDs_dir, 'DoD_' + run, DoD_name)
# Load file:
DoD = np.loadtxt(DoD_path, delimiter='\t')
# Convert NaN to np.nan
matrix = np.where(DoD==-999, np.nan, DoD)


def morph_quantities(matrix):
    import numpy as np
    '''
    
    Parameters
    ----------
    matrix : 2D numpy matrix
        DESCRIPTION.

    Returns
    -------
    Volumes are calculated as different sum of the cells values so they appear
    as a length. To obtain the "real" volume, the length values must be multiply
    by the cell dimension as: V=tot_vol*dim_x*dim_y
    tot_vol : real
        Total net volume as the algebric sum of all the cells [L]
    sum_vol : real
        Sum of scour and deposition volume as algebric sum of the abs of each cell [L]
    dep_vol : real
        Deposition volume as the sum of the value of deposition cell  [L]
    sco_vol : real
        Scour volume as the sum of the value of scour cell [L]
        
    Active area array is calculated as the number of DoD active cell.
    morph_act_area : integer
        Active area both in terms of scour and deposition in number of cells [-]
    morph_act_area_dep : integer
        Active deposition area in number of cells [-]
    morph_act_area_sco : integer
        Active scour area in number of cells [-]
        
    The active width is calculated as the active area divided by the DoD length
    in terms of number of cells. So the active width is given in number of cell
    and to obtain the width dimension the data need to be multiply by dim_y
    act_width_mean : real
        Total mean active width in number of cells [-]
    act_width_mean_dep : real
        Deposition mean active width in number of cells [-]
    act_width_mean_sco : real
        Scour mean active width in number of cells [-]
        
    act_thickness :
        Active thickness as the average of scour and deposition active thickness [L]
    act_thickness_dep :
        Deposition active thickness (abs(V_sco) + V_dep)/act_area [L]
    act_thickness_sco :
        Scour active thickness (abs(V_sco) + V_dep)/act_area [L]
    bri :
        Bed Relief Index calculated as the standard deviation of the DoD values [L]

    
    This function ...

    '''
    
    # Define total volume matrix, Deposition matrix and Scour matrix
    vol_array = np.where(np.isnan(matrix), 0, matrix) # Total volume matrix
    dep_array = (vol_array>0)*vol_array # DoD of only deposition data
    sco_array = (vol_array<0)*vol_array # DoD of only scour data
    
    # Volume are calculated as the sum of the cell value. The measure unit is a length.
    # To obtain a volume, the _vol value has to be multiply by the cell dimension.
    tot_vol = np.sum(vol_array) # Total net volume as the algebric sum of all the cells [L]
    sum_vol = np.sum(np.abs(vol_array)) # Sum of scour and deposition volume as algebric sum of the abs of each cell [L]
    dep_vol = np.sum(dep_array) # Deposition volume as the sum of the value of deposition cell  [L]
    sco_vol = np.sum(sco_array) # Scour volume as the sum of the value of scour cell [L]
    
    # Define nature matrix as -1=sco, 0=no_changes, and 1=dep
    nature_array = np.where(matrix>0, 1, matrix)
    nature_array = np.where(nature_array<0, -1, nature_array)
    
    # Define activity array: VERIFIED
    tot_act_array = np.where(np.isnan(nature_array), 0, nature_array) # Where active then 1
    dep_act_array = tot_act_array*(tot_act_array>0) # Where scour then 1
    sco_act_array = tot_act_array*(tot_act_array<0) # Where scour then 1
    
    # Calculate morphological quantities VERIFIED
    # Active area array is calculated as the number of active cell. To obtain a width the number of cell has to be multiply by the crosswise length of the generic cell
    morph_act_area = np.count_nonzero(abs(tot_act_array)) # Active area both in terms of scour and deposition in number of cells [-]
    morph_act_area_dep = np.sum(dep_act_array) # Active deposition area in number of cells [-]
    morph_act_area_sco = np.sum(abs(sco_act_array)) # Active scour area in number of cells [-]
    
    # Create active width for each cross section
    act_width_array = np.array([np.nansum(abs(tot_act_array), axis=0)]) # Array of the crosswise morphological total active width in number of cells
    act_width_array_dep = np.array([np.nansum(dep_act_array, axis=0)]) # Array of the crosswise morphological deposition active width in number of cells
    act_width_array_sco = np.array([np.nansum(abs(sco_act_array), axis=0)]) # Array of the crosswise morphological scour active width in number of cells
    
    # Calculate the mean of each active width array: VERIFIED
    act_width_mean = np.nanmean(act_width_array) # Total mean active width in number of cells (could be a real number)
    act_width_mean_dep = np.nanmean(act_width_array_dep) # Deposition mean active width in number of cells (could be a real number)
    act_width_mean_sco = np.nanmean(act_width_array_sco) # Scour mean active width in number of cells (could be a real number)
    
    # Calculate active thickness for total volumes, deposition volumes and scour volumes VERIFIED
    vol_array=np.where(vol_array==0, np.nan, vol_array)
    dep_array=np.where(dep_array==0, np.nan, dep_array)
    sco_array=np.where(sco_array==0, np.nan, sco_array)
    act_thickness = np.nanmean(np.abs(dep_array)) + np.nanmean(np.abs(sco_array)) # Active thickness as the average of scour and deposition active thickness
    act_thickness_dep = np.nanmean(np.abs(dep_array)) # Deposition active thickness (abs(V_sco) + V_dep)/act_area [mm]
    act_thickness_sco = np.nanmean(np.abs(sco_array)) # Scour active thickness (abs(V_sco) + V_dep)/act_area [mm]
    
    # Calculate the Bed Relief Index
    bri = np.nanstd(matrix)
    
    return tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri

tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(matrix)