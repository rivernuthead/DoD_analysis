#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:45:38 2022

@author: erri
"""

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
    
    # Define activity array: VERIFIED
    tot_act_array = np.where(np.isnan(nature_array), 0, nature_array) # Where active then 1
    dep_act_array = tot_act_array*(tot_act_array>0) # Where scour then 1
    sco_act_array = tot_act_array*(tot_act_array<0) # Where scour then 1
    
    # Calculate morphological quantities VERIFIED
    morph_act_area = np.count_nonzero(tot_act_array) # Active area both in terms of scour and deposition in number of cells [-]
    morph_act_area_dep = np.sum(dep_act_array) # Active deposition area in number of cells [-]
    morph_act_area_sco = np.sum(sco_act_array) # Active scour area in number of cells [-]
    
    # Create active width for each cross section
    act_width_array = np.array([np.nansum(tot_act_array, axis=0)]) # Array of the crosswise morphological total active width in number of cells
    act_width_array_dep = np.array([np.nansum(dep_act_array, axis=0)]) # Array of the crosswise morphological deposition active width in number of cells
    act_width_array_sco = np.array([np.nansum(sco_act_array, axis=0)]) # Array of the crosswise morphological scour active width in number of cells
    
    # Calculate the mean of each active width array: VERIFIED
    act_width_mean = np.nanmean(act_width_array) # Total mean active width in number of cells (could be a real number)
    act_width_mean_dep = np.nanmean(act_width_array_dep) # Deposition mean active width in number of cells (could be a real number)
    act_width_mean_sco = np.nanmean(act_width_array_sco) # Scour mean active width in number of cells (could be a real number)
    
    # Calculate active thickness for total volumes, deposition volumes and scour volumes VERIFIED
    vol_array=np.where(vol_array==0, np.nan, vol_array)
    dep_array=np.where(dep_array==0, np.nan, dep_array)
    sco_array=np.where(sco_array==0, np.nan, sco_array)
    act_thickness = np.nanmean(np.abs(vol_array)) # Active thickness as the average of scour and deposition active thickness
    act_thickness_dep = np.nanmean(np.abs(dep_array)) # Deposition active thickness (abs(V_sco) + V_dep)/act_area [mm]
    act_thickness_sco = np.nanmean(np.abs(sco_array)) # Scour active thickness (abs(V_sco) + V_dep)/act_area [mm]
    
    return tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco