#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
# from DoD_analysis_functions import *
# from DoD_analysis_functions_4 import *
from morph_quantities_func_v2 import morph_quantities
from matplotlib.colors import ListedColormap, BoundaryNorm

###############################################################################
# TODO
###############################################################################
# 1. 

start = time.time() # Set initial time

###############################################################################
# SETUP SCRIPT PARAMETERS and RUN MODE
###############################################################################

'''
run mode:
    0 = runs in the runs list
    1 = one run at time
    2 = bath process 'all the runs in the folder'
data_interpolatuon_mode:
    0 = no interpolation
    1 = data interpolation
windows_mode:
    0 = fixed windows (all the channel)
    1 = floating windows
    2 = fixed windows (WxW, Wx2W, Wx3W, ...) without overlapping
    3 = fixed windows (WxW, Wx2W, Wx3W, ...) with overlapping
mask mode:
    1 = mask the flume edge
    2 = mask the upstream half flume
    3 = mask the downstream half flume
process mode: (NB: set DEMs name)
    1 = batch process
    2 = single run process
save mode:
    0 = save only reports
    1 = save all chart and figure
DoD_plot_save_mode:
    0 = do not save DoD plots
    1 = save plot
DoD_plot_show_mode:
    0 = do not show DoD plots
    1 = show DoD plots
'''
run_mode = 0
mask_mode = 1
process_mode = 1
# data_interpolation_mode = 0
# save_plot_mode = 1
# DoD_plot_mode = 0
# DoD_plot_save_mode = 1
# DoD_plot_show_mode = 0

# SINGLE RUN NAME
run = 'q15_3'
# ARRAY OF RUNS
runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']

# Set DEM single name to perform process to specific DEM
if len(run) ==1:
    DEM1_single_name = 'matrix_bed_norm_' + str(run) +'s'+'0'+'.txt' # DEM1 name
    DEM2_single_name = 'matrix_bed_norm_' + str(run) +'s'+'1'+'.txt' # DEM2 name

# Filtering process thresholds values
# thrs_zeros = 7 # [-] isolated_killer function threshold
# thrs_nature = 5 # [-] nature_checker function threshold
# thrs_fill = 7 # [-] isolated_filler function threshold
# thrs_1 = 2.0  # [mm] # Lower threshold
# thrs_2 = 15.0  # [mm] # Upper threshold

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

#%%
###############################################################################
# SETUP FOLDERS and RUNS
###############################################################################
# setup working directory and DEM's name
home_dir = os.getcwd()
out_dir = os.path.join(home_dir, 'output')
plot_out_dir = os.path.join(home_dir, 'plot')

# Create folders
if not(os.path.exists(out_dir)):
    os.mkdir(out_dir)
if not(os.path.exists(plot_out_dir)):
    os.mkdir(plot_out_dir)
    
DoDs_dir = os.path.join(home_dir, 'DoDs')

run_dir = os.path.join(home_dir, 'surveys')



# Create the run name list
RUNS=[]
if run_mode==0:
    RUNS=runs
elif run_mode==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run.split() # RUNS as a single entry array, provided by run variable

#%%
###############################################################################
# INITIALIZE OVERALL REPORT ARRAYS
###############################################################################

# Define volume time scale report matrix:
# B_dep, SD(B_dep), B_sco, SD(B_sco)
volume_temp_scale_report=np.zeros((len(RUNS), 4))

# Define morphW time scale report matrix:
# B_morphW [min], SD(B_morphW)
morphW_temp_scale_report = np.zeros((len(RUNS), 2))

# # Define Engelund Gauss model report matrix:
# # D [m], Q [m^3/s], Wwet/W [-]
# engelund_model_report=np.zeros((len(RUNS),3))

# Array that collect all the morphWact_array dimension.
# It will be used to create the morphWact_matrix
morphWact_dim = [] # Array with the dimensions of morphWact_values array

DoD_length_array=[] # DoD length array

#%%
###############################################################################
# MAIN LOOP OVER RUNS
###############################################################################
for run in RUNS:

    ###########################################################################
    # SETUP FOLDERS
    ###########################################################################
    print('######')
    print(run)
    print('######')
    print()
    # setup working directory and DEM's name
    input_dir = os.path.join(home_dir, 'DoDs', 'DoDs_' + run)
    report_dir = os.path.join(home_dir, 'output', run)
    plot_dir = os.path.join(home_dir, 'plot', run)
    
    # Save a report with xData as real time in minutes and the value of scour and deposition volumes for each runs
    # Check if the file already exists
    if os.path.exists(os.path.join(report_dir, 'volume_over_time.txt')):
        os.remove(os.path.join(report_dir, 'volume_over_time.txt'))
    else:
        pass

    # CREATE FOLDERS
    if not(os.path.exists(report_dir)):
        os.mkdir(report_dir)
    if not(os.path.exists(DoDs_dir)):
        os.mkdir(DoDs_dir)
    if not(os.path.exists(os.path.join(DoDs_dir, 'DoDs_stack'))):
        os.mkdir(os.path.join(DoDs_dir, 'DoDs_stack'))
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)


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
    
    # Flume geometry parameters
    W = run_param[0,4] # Flume width [m]
    S = run_param[0,5] # Flume slope
    
    # Sediment parameters
    d50 = run_param[0,6]

    # Run discharge
    Q = run_param[0,0] # Run discharge [l/s]
    
    # Create the list of DoD files and sort them in the files list.
    # To do so, two system array will be created for the run with more than 10 surveys due to the behaviour of the sort python function that sort files in this way:
    # matrix_bed_norm_q10_3s0.txt matrix_bed_norm_q10_3s1.txt matrix_bed_norm_q10_3s11.txt matrix_bed_norm_q10_3s12.txt matrix_bed_norm_q10_3s2.txt
    files = []
    files1=[] # Files list with surveys from 0 to 9
    files2=[] # Files list with surveys from 10
    files3=[]
    # Creating array with file names:
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and f.startswith('DoD_') and f.endswith('ult_gis.txt'):
            files = np.append(files, f)
    for f in files:
        if len(f)==len(files[1])-1:
            files1 = np.append(files1, f)
        elif len(f)==len(files[1]):
            files2 = np.append(files2, f)
        elif len(f)==len(files[1])+1:
            files3 = np.append(files3,f)
            
    files = np.append(files1,files2) # Files has been overwritten with a list of file names sorted in the right way :) 
    files = np.append(files,files3)
    
    # Calculate the dimension of the report matrix:
    # If the number of the file list is M, it's true that n*(n+1)/2=M where M is the number of DoD spaced 1 timestep
    # The equation can be resolved as: n=(-1+sqrt(1-4*(-78*2)))/(2)
    
    n = int((-1+np.sqrt(1-4*(-len(files)*2)))/(2)) +1
    
    # INITIALIZE ARRAYS
    comb = np.array([]) # combination of differences
    DoD_act_px_count_array=[] # Array where collect all the DoDs active pixel counting
    volumes_array=[] # Tot volume
    dep_array=[] # Deposition volume
    sco_array=[] # Scour volume
    sum_array=[] # Sum of scour and deposition volume
    morph_act_area_array=[] # Total active area array
    morph_act_area_array_dep=[] # Deposition active area array
    morph_act_area_array_sco=[] # Active active area array
    act_width_mean_array=[] # Total active width mean array
    act_width_mean_array_dep=[] # Deposition active width mean array
    act_width_mean_array_sco=[] # Scour active width mean array
    morphWact_values=[] # morphWact values for each section of all the DoD
    morphWact_values_dep=[] # morphWact values for each section of all the DoD
    morphWact_values_sco=[] # morphWact values for each section of all the DoD
    report_matrix = [] #Report matrix
    # matrix_volumes=np.zeros((n-1, n+1)) # Volumes report matrix
    matrix_volumes=np.zeros((n-1, n+1)) # Volumes report matrix
    matrix_sum_volumes=np.zeros((n-1, n+1)) # Sum of scour and deposition volumes
    # matrix_dep=np.zeros((n-1, n+1)) # Deposition volume report matrix
    matrix_dep=np.zeros((n+3, n+1)) # Deposition volume report matrix
    matrix_morph_act_area=np.zeros((n+3, n+1)) # Total active area report matrix
    matrix_morph_act_area_dep=np.zeros((n+3, n+1)) # Deposition active area report matrix
    matrix_morph_act_area_sco=np.zeros((n+3, n+1)) # Scour active area report matrix
    # matrix_sco=np.zeros((n-1, n+1)) # Scour volume report matrix
    matrix_sco=np.zeros((n+3, n+1)) # Scour volume report matrix
    matrix_Wact=np.zeros((n+3, n+3)) # Active width report matrix
    matrix_Wact_sco=np.zeros((n+3, n+3)) # Active width report matrix
    matrix_Wact_dep=np.zeros((n+3, n+3)) # Active width report matrix
    matrix_Wact_IIIquantile=np.zeros((n-1, n+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile=np.zeros((n-1, n+1)) # I quantile active width report matrix
    matrix_Wact_IIIquantile_dep=np.zeros((n-1, n+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile_dep=np.zeros((n-1, n+1)) # I quantile active width report matrix
    matrix_Wact_IIIquantile_sco=np.zeros((n-1, n+1)) # III quantile active width report matrix
    matrix_Wact_Iquantile_sco=np.zeros((n-1, n+1)) # I quantile active width report matrix
    matrix_act_thickness = np.zeros((n-1, n+1)) # Matrix where collect total active thickness data
    matrix_act_thickness_dep = np.zeros((n-1, n+1)) # Matrix where collect deposition active thickness data
    matrix_act_thickness_sco = np.zeros((n-1, n+1)) # Matrix where collect scour active thickness data
    matrix_act_volume = np.zeros((n-1, n+1)) # Matrix where collect volume data

    matrix_DEM_analysis = np.zeros((n, n))
    
    # Analysis on the effect of the spatial filter on the morphological changes
    # Initialize array
    DoD_raw_morph_quant = []
    DoD_filt_mean_morph_quant = []
    DoD_filt_isol_morph_quant = []
    DoD_filt_fill_morph_quant = []
    DoD_filt_nature_morph_quant = []
    DoD_filt_isol2_morph_quant = []
    DoD_filt_ult_morph_quant = []

    ###########################################################################
    # CHECK DEMs SHAPE
    ###########################################################################
    # Due to differences between DEMs shape (not the same ScanArea.txt laser survey file)
    # a preliminary loop over the all DEMs is required in order to define the target
    # dimension of the reshaping operation
    array_dim_x = []
    array_dim_y = []
    for f in sorted(files):
        path_DoD = os.path.join(input_dir, f)
        DoD = np.loadtxt(path_DoD,
                          # delimiter=',',
                          skiprows=8
                          )
        
        array_dim_x = np.append(array_dim_x, DoD.shape[0])
        array_dim_y = np.append(array_dim_y, DoD.shape[1])

    # Define target dimension:
    shp_target_x, shp_target_y = int(min(array_dim_x)), int(min(array_dim_y))

    arr_shape = np.array([shp_target_x, shp_target_y]) # Define target shape



    ###########################################################################
    # LOOP OVER ALL DEMs COMBINATIONS
    ###########################################################################
    # Perform difference between DEMs over all the possible combination of surveys in the survey directory
    for i in range(0,len(files)):
        f=files[i]
        DoD_path = os.path.join(input_dir, f)
        print(DoD_path)
        DoD = np.loadtxt(DoD_path, delimiter='\t', skiprows=8)
        DoD = np.where(DoD==NaN, np.nan, DoD)
        
        comb = np.append(comb, f) # Create a list with all the available combinations of DEMs

        
        print(run)
        print('=========')
        print(f)
        
        
        # DEFINE THE DoD NUMBER ACCORDING TO THE NUMBER OF THE DEM
        # 20 is the number of character of the shorter DoD name
        if len(f)==24: # DoD_1-0_filt_ult.txt    20
            DEM1_num = int(f[-18:-17])
            DEM2_num = int(f[-20:-19])
        elif len(f)==24+1: # DoD_10-0_filt_ult.txt    21
            DEM1_num = int(f[-18:-17])
            DEM2_num = int(f[-21:-19])
        elif len(f)==24+2: # DoD_12-11_filt_ult.txt    22
            DEM1_num = int(f[-19:-17])
            DEM2_num = int(f[-22:-20])
        else:
            print('Error in the definition of the DoD name')
            break

        print('=========')
        print(DEM2_num, ' ', DEM1_num)
            
            
        
            
        delta = int(DEM2_num) - int(DEM1_num)
        
        print('delta = ', delta)
        print('----------')
        print()
        
        
        # Raster dimension
        dim_y, dim_x = DoD.shape
        print('dim_x: ', dim_x, '    dim_y: ', dim_y)
        

        # Calculate the DoD length in meters:
        DoD_length = DoD.shape[1]*px_x/1000 # DoD length [m]
        


        # Count the number of pixels in the channel area
        DoD_count = np.count_nonzero(np.where(np.isnan(DoD), 0, 1))
        print('Number of channel pixel pixels:', DoD_count)
        
        # Append for each DoD the number of active pixels to the DoD_act_px_count_array
        DoD_act_px_count_array = np.append(DoD_act_px_count_array, DoD_count)
           
        
        DoD_filt_ult = DoD

        ###################################################################
        # TOTAL VOLUMES, DEPOSITION VOLUMES AND SCOUR VOLUMES
        ###################################################################
        # DoD filtered name: DoD_filt
        # Create new raster where apply volume calculation
        # DoD>0 --> Deposition, DoD<0 --> Scour
        # =+SUMIFS(A1:JS144, A1:JS144,">0")*5*50(LibreCalc function)
        
        # SCOUR AND DEPOSITION MATRIX, DEPOSITION ONLY MATRIX AND SCOUR ONLY MATRIX:
        DoD_vol = np.where(np.isnan(DoD_filt_ult), 0, DoD_filt_ult) # Total volume matrix
        dep_DoD = (DoD_vol>0)*DoD_vol # Deposition only matrix
        sco_DoD = (DoD_vol<0)*DoD_vol # Scour only matrix
        
        # ...as boolean active pixel matrix:
        act_px_matrix = np.where(DoD_filt_ult!=0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active pixel matrix, both scour and deposition
        act_px_matrix_dep = np.where(DoD_filt_ult>0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active deposition matrix 
        act_px_matrix_sco = np.where(DoD_filt_ult<0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active scour matrix
        
        # GIS readable matrix where np.nan is NaN
        act_px_matrix_gis = np.where(np.isnan(act_px_matrix), NaN, act_px_matrix) # Active pixel matrix, both scour and deposition
        act_px_matrix_dep_gis = np.where(np.isnan(act_px_matrix_dep), NaN, act_px_matrix_dep) # Active deposition matrix 
        act_px_matrix_sco_gis = np.where(np.isnan(act_px_matrix_sco), NaN, act_px_matrix_sco) # Active scour matrix
        
        # MORPHOLOGICAL QUANTITIES:
        tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
        
        DoD_name = 'DoD_' + str(DEM2_num) + '-' + str(DEM1_num)
        DoD_raw = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_raw.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_mean = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_mean.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_isol = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_isol.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_nature = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_nature.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_fill = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_fill.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_isol2 = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_isol2.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_ult = np.loadtxt(os.path.join(DoDs_dir,'DoDs_'+run, DoD_name + '_filt_ult.txt'),skiprows=8 , delimiter='\t')
        
        # Convert all zero value in np.nan to make it transparent on plots:
        DoD_raw_plot = np.where(DoD_raw==0, np.nan, DoD_raw)
        DoD_filt_mean_plot = np.array(np.where(DoD_filt_mean==0, np.nan, DoD_filt_mean))
        DoD_filt_isol_plot = np.array(np.where(DoD_filt_isol==0, np.nan, DoD_filt_isol))
        DoD_filt_nature_plot = np.array(np.where(DoD_filt_nature==0, np.nan, DoD_filt_nature))
        DoD_filt_fill_plot = np.array(np.where(DoD_filt_fill==0, np.nan, DoD_filt_fill))
        DoD_filt_isol2_plot = np.array(np.where(DoD_filt_isol2==0, np.nan, DoD_filt_isol2))
        DoD_filt_ult_plot = np.array(np.where(DoD_filt_ult==0, np.nan, DoD_filt_ult))
        
        
        # Analysis to investigate the role of the application of spatial filters at the DoD
        # in the morphological changes calculation
        if delta==1:
            # DoD_raw
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_raw)
            if len(DoD_raw_morph_quant)==0:
                DoD_raw_morph_quant=np.append(DoD_raw_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_raw_morph_quant=np.vstack((DoD_raw_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_mean
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_mean)
            if len(DoD_filt_mean_morph_quant)==0:
                DoD_filt_mean_morph_quant=np.append(DoD_filt_mean_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_mean_morph_quant=np.vstack((DoD_filt_mean_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_isol
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol)
            if len(DoD_filt_isol_morph_quant)==0:
                DoD_filt_isol_morph_quant=np.append(DoD_filt_isol_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_isol_morph_quant=np.vstack((DoD_filt_isol_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_fill
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_fill)
            if len(DoD_filt_fill_morph_quant)==0:
                DoD_filt_fill_morph_quant=np.append(DoD_filt_fill_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_fill_morph_quant=np.vstack((DoD_filt_fill_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_nature
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_nature)
            if len(DoD_filt_nature_morph_quant)==0:
                DoD_filt_nature_morph_quant=np.append(DoD_filt_nature_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_nature_morph_quant=np.vstack((DoD_filt_nature_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_isol2
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol2)
            if len(DoD_filt_isol2_morph_quant)==0:
                DoD_filt_isol2_morph_quant=np.append(DoD_filt_isol2_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_isol2_morph_quant=np.vstack((DoD_filt_isol2_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
            # DoD_filt_ult
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
            if len(DoD_filt_ult_morph_quant)==0:
                DoD_filt_ult_morph_quant=np.append(DoD_filt_ult_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
            else:
                DoD_filt_ult_morph_quant=np.vstack((DoD_filt_ult_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
            
        #Print results:
        print('Total volume:', "{:.1f}".format(tot_vol))
        print('Sum of deposition and scour volume:', "{:.1f}".format(sum_vol))
        print('Deposition volume:', "{:.1f}".format(dep_vol))
        print('Scour volume:', "{:.1f}".format(sco_vol))

        # Append values to output data array
        volumes_array = np.append(volumes_array, tot_vol)
        sum_array = np.append(sum_array, sum_vol)
        dep_array = np.append(dep_array, dep_vol)
        sco_array = np.append(sco_array, sco_vol)
        
        
        
        ###################################################################
        # Active_pixel analysis
        ###################################################################
        
        # Morphological active area
        morph_act_area_array = np.append(morph_act_area_array, morph_act_area) # For each DoD, append total active area data
        morph_act_area_array_dep = np.append(morph_act_area_array_dep, morph_act_area_dep) # For each DoD, append deposition active area data
        morph_act_area_array_sco = np.append(morph_act_area_array_sco, morph_act_area_sco) # For each DoD, append scour active area data
        
        # Morphological active width streamwise array [number of activ cells]
        act_width_array = np.array([np.nansum(act_px_matrix, axis=0)])/(W/(px_y/1000)) # Array of the crosswise morphological total active width in number of active cells [-]
        act_width_array_dep = np.array([np.nansum(act_px_matrix_dep, axis=0)])/(W/(px_y/1000)) # Array of the crosswise morphological deposition active width in number of active cells [-]
        act_width_array_sco = np.array([np.nansum(act_px_matrix_sco, axis=0)])/(W/(px_y/1000)) # Array of the crosswise morphological scour active width in number of active cells [-]
        
        
        # Morphological active width [mean value, number of activ cells / channel width]
        act_width_mean_array = np.append(act_width_mean_array, act_width_mean/(W/(px_y/1000))) # For each DoD append total active width values [actW/W]
        act_width_mean_array_dep = np.append(act_width_mean_array_dep, act_width_mean_dep/(W/(px_y/1000))) # For each DoD append deposition active width values [actW/W]
        act_width_mean_array_sco = np.append(act_width_mean_array_sco, act_width_mean_sco/(W/(px_y/1000))) # For each DoD append scour active width values [actW/W]
        
        
        
        print('Active thickness [mm]:', act_thickness)
        print('Morphological active area (number of active cells): ', "{:.1f}".format(morph_act_area), '[-]')
        print('Morphological active width (mean):', "{:.3f}".format(act_width_mean/(W/(px_y/1000))), 'actW/W [-]')
        print()
        
        # Create output matrix as below:
        # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
        # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
        # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
        # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
        # DoD step4  5-0   6-1   7-2   8-3   9-4                          average STDEV
        # DoD step5  6-0   7-1   8-2   9-3                                average STDEV
        # DoD step6  7-0   8-1   9-2                                      average STDEV
        # DoD step7  8-0   9-1                                            average STDEV
        # DoD step8  9-0                                                  average STDEV
        #             A     A     A     A     A     A     A     A     A
        #           SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A) SD(A)
        #             B     B     B     B     B     B     B     B     B
        #           SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B) SD(B)

        # # TODO UPDATE
        # delta=int(DEM2_num)-int(DEM1_num) # Calculate delta between DEM
        
        # print('delta = ', delta)
        # print('----------')
        # print()
        # print()
        
        # Build up morphWact/W array for the current run boxplot
        # This array contain all the morphWact/W values for all the run repetition in the same line
        # This array contain only adjacent DEMs DoD
        if delta==1:
            morphWact_values = np.append(morphWact_values, act_width_array)
            morphWact_values_dep = np.append(morphWact_values_dep, act_width_array_dep)
            morphWact_values_sco = np.append(morphWact_values_sco, act_width_array_sco)

        
        # Fill Scour, Deposition and morphWact/w matrix:
            
        # Fill matrix with data
        matrix_volumes[delta-1,DEM1_num]=tot_vol # Total volumes as the algebric sum of scour and deposition volumes [L]
        matrix_sum_volumes[delta-1,DEM1_num]=sum_vol # Total volumes as the sum of scour and deposition volumes [L]
        
        matrix_dep[delta-1,DEM1_num]=dep_vol # Deposition volume [L]
        
        matrix_sco[delta-1,DEM1_num]=sco_vol # Scour volume [L]
        matrix_morph_act_area[delta-1,DEM1_num]=morph_act_area # Total morphological active area in number of cells [-]
        matrix_morph_act_area_dep[delta-1,DEM1_num]=morph_act_area_dep # Deposition morphological active area in number of cells [-]
        matrix_morph_act_area_sco[delta-1,DEM1_num]=morph_act_area_sco # Scour morphological active area in number of cells [-]
        matrix_act_thickness[delta-1,DEM1_num]=act_thickness # Active thickness data calculated from total volume matrix [L]
        matrix_act_thickness_dep[delta-1,DEM1_num]=act_thickness_dep # Active thickness data calculated from deposition volume matrix [L]
        matrix_act_thickness_sco[delta-1,DEM1_num]=act_thickness_sco # Active thickness data calculated from scour volume matrix [L]
        matrix_Wact[delta-1,DEM1_num]=act_width_mean/(W/(px_y/1000))
        matrix_Wact_dep[delta-1,DEM1_num]=act_width_mean_dep/(W/(px_y/1000))
        matrix_Wact_sco[delta-1,DEM1_num]=act_width_mean_sco/(W/(px_y/1000))
        
        # Fill last two columns with AVERAGE of the corresponding row
        matrix_volumes[delta-1,-2]=np.average(matrix_volumes[delta-1,:n-delta]) #Total volumes
        matrix_sum_volumes[delta-1,-2]=np.average(matrix_sum_volumes[delta-1,:n-delta]) #Total sum volumes
        matrix_dep[delta-1,-2]=np.average(matrix_dep[delta-1,:n-delta]) # Deposition volumes
        matrix_sco[delta-1,-2]=np.average(matrix_sco[delta-1,:n-delta]) # Scour volumes
        matrix_morph_act_area[delta-1,-2]=np.average(matrix_morph_act_area[delta-1,:n-delta]) # Morphological total active area
        matrix_morph_act_area_dep[delta-1,-2]=np.average(matrix_morph_act_area_dep[delta-1,:n-delta]) # Morphological deposition active area
        matrix_morph_act_area_sco[delta-1,-2]=np.average(matrix_morph_act_area_sco[delta-1,:n-delta]) # Morphological scour active area
        matrix_act_thickness[delta-1,-2]=np.average(matrix_act_thickness[delta-1,:n-delta]) # Fill matrix with active thickness average calculated from total volume matrix
        matrix_act_thickness_dep[delta-1,-2]=np.average(matrix_act_thickness_dep[delta-1,:n-delta]) # Active thickness average calculated from deposition volume matrix
        matrix_act_thickness_sco[delta-1,-2]=np.average(matrix_act_thickness_sco[delta-1,:n-delta]) # Active thickness average calculated from scour volume matrix
        matrix_Wact[delta-1,-2]=np.average(matrix_Wact[delta-1,:n-delta])
        matrix_Wact_dep[delta-1,-2]=np.average(matrix_Wact_dep[delta-1,:n-delta])
        matrix_Wact_sco[delta-1,-2]=np.average(matrix_Wact_sco[delta-1,:n-delta])
        
        # Fill last two columns with STDEV of the corresponding row
        matrix_volumes[delta-1,-1]=np.std(matrix_volumes[delta-1,:n-delta])
        matrix_sum_volumes[delta-1,-1]=np.std(matrix_sum_volumes[delta-1,:n-delta])
        matrix_dep[delta-1,-1]=np.std(matrix_dep[delta-1,:n-delta])
        matrix_sco[delta-1,-1]=np.std(matrix_sco[delta-1,:n-delta])
        matrix_morph_act_area[delta-1,-1]=np.std(matrix_morph_act_area[delta-1,:n-delta])
        matrix_morph_act_area_dep[delta-1,-1]=np.std(matrix_morph_act_area_dep[delta-1,:n-delta])
        matrix_morph_act_area_sco[delta-1,-1]=np.std(matrix_morph_act_area_sco[delta-1,:n-delta])
        matrix_act_thickness[delta-1,-1]=np.std(matrix_act_thickness[delta-1,:n-delta]) # Fill matrix with active thickness standard deviation calculated from total volume matrix
        matrix_act_thickness_dep[delta-1,-1]=np.std(matrix_act_thickness_dep[delta-1,:n-delta]) # Active thickness average calculated from deposition volume matrix
        matrix_act_thickness_sco[delta-1,-1]=np.std(matrix_act_thickness_sco[delta-1,:n-delta]) # Active thickness average calculated from scour volume matrix
        matrix_Wact[delta-1,-1]=np.std(matrix_Wact[delta-1,:n-delta])
        matrix_Wact_dep[delta-1,-1]=np.std(matrix_Wact_dep[delta-1,:n-delta])
        matrix_Wact_sco[delta-1,-1]=np.std(matrix_Wact_sco[delta-1,:n-delta])
        
        # Fill III quantile Wact/W matrix:
        matrix_Wact_IIIquantile[delta-1,DEM1_num]=np.quantile(act_width_array, .75)
        matrix_Wact_IIIquantile[delta-1,-2]=np.min(matrix_Wact_IIIquantile[delta-1,:n-delta])
        matrix_Wact_IIIquantile[delta-1,-1]=np.max(matrix_Wact_IIIquantile[delta-1,:n-delta])
        
        matrix_Wact_IIIquantile_dep[delta-1,DEM1_num]=np.quantile(act_width_array_dep, .75)
        matrix_Wact_IIIquantile_dep[delta-1,-2]=np.min(matrix_Wact_IIIquantile_dep[delta-1,:n-delta])
        matrix_Wact_IIIquantile_dep[delta-1,-1]=np.max(matrix_Wact_IIIquantile_dep[delta-1,:n-delta])
        
        matrix_Wact_IIIquantile_sco[delta-1,DEM1_num]=np.quantile(act_width_array_sco, .75)
        matrix_Wact_IIIquantile_sco[delta-1,-2]=np.min(matrix_Wact_IIIquantile_sco[delta-1,:n-delta])
        matrix_Wact_IIIquantile_sco[delta-1,-1]=np.max(matrix_Wact_IIIquantile_sco[delta-1,:n-delta])


        # Fill I quantile Wact/W matrix:
        matrix_Wact_Iquantile[delta-1,DEM1_num]=np.quantile(act_width_array, .25)
        matrix_Wact_Iquantile[delta-1,-2]=np.min(matrix_Wact_Iquantile[delta-1,:n-delta])
        matrix_Wact_Iquantile[delta-1,-1]=np.max(matrix_Wact_Iquantile[delta-1,:n-delta])
        
        matrix_Wact_Iquantile_dep[delta-1,DEM1_num]=np.quantile(act_width_array_dep, .25)
        matrix_Wact_Iquantile_dep[delta-1,-2]=np.min(matrix_Wact_Iquantile_dep[delta-1,:n-delta])
        matrix_Wact_Iquantile_dep[delta-1,-1]=np.max(matrix_Wact_Iquantile_dep[delta-1,:n-delta]) 
        
        matrix_Wact_Iquantile_sco[delta-1,DEM1_num]=np.quantile(act_width_array_sco, .25)
        matrix_Wact_Iquantile_sco[delta-1,-2]=np.min(matrix_Wact_Iquantile_sco[delta-1,:n-delta])
        matrix_Wact_Iquantile_sco[delta-1,-1]=np.max(matrix_Wact_Iquantile_sco[delta-1,:n-delta]) 
        

        # DATA STRUCTURE
        # Fill Wact/W MEAN matrix as below:
        # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  Iquantile IIIquantile average STDEV
        # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        Iquantile IIIquantile average STDEV
        # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              Iquantile IIIquantile average STDEV
        # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    Iquantile IIIquantile average STDEV
        # DoD step4  5-0   6-1   7-2   8-3   9-4                          Iquantile IIIquantile average STDEV
        # DoD step5  6-0   7-1   8-2   9-3                                Iquantile IIIquantile average STDEV
        # DoD step6  7-0   8-1   9-2                                      Iquantile IIIquantile average STDEV
        # DoD step7  8-0   9-1                                            Iquantile IIIquantile average STDEV
        # DoD step8  9-0                                                  Iquantile IIIquantile average STDEV

        # Fill Wact/W MAX (MIN) matrix as below:
        # NB: MIN and MAX columns are to be intended as the maximum and the minimum value
        # of the IIIquantile (or Iquantile) values of DoDs row. So the MIN value of the
        # matrix_Wact_IIIquantile is the minimum value between the maximum value.
        # DoD step0  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  min(Iquantile) max(Iquantile)
        # DoD step1  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        min(Iquantile) max(Iquantile)
        # DoD step2  3-0   4-1   5-2   6-3   7-4   8-5   9-6              min(Iquantile) max(Iquantile)
        # DoD step3  4-0   5-1   6-2   7-3   8-4   9-5                    min(Iquantile) max(Iquantile)
        # DoD step4  5-0   6-1   7-2   8-3   9-4                          min(Iquantile) max(Iquantile)
        # DoD step5  6-0   7-1   8-2   9-3                                min(Iquantile) max(Iquantile)
        # DoD step6  7-0   8-1   9-2                                      min(Iquantile) max(Iquantile)
        # DoD step7  8-0   9-1                                            min(Iquantile) max(Iquantile)
        # DoD step8  9-0                                                  min(Iquantile) max(Iquantile)
        


    # Fill DoD lenght array
    DoD_length_array = np.append(DoD_length_array, DoD_length)



    ###############################################################################
    # SAVE DATA MATRIX
    ###############################################################################
    # Create report matrix
    report_matrix = np.array(np.transpose(np.stack((comb, DoD_act_px_count_array, volumes_array, dep_array, sco_array, morph_act_area_array, act_width_mean_array))))
    report_header = 'DoD_combination, Active pixels, Total volume [mm^3], Deposition volume [mm^3], Scour volume [mm^3], Active area [mm^2], Active width mean [%]'

    report_name = run + '_report.txt'
    with open(os.path.join(report_dir , report_name), 'w') as fp:
        fp.write(report_header)
        fp.writelines(['\n'])
        for i in range(0,len(report_matrix[:,0])):
            for j in range(0, len(report_matrix[0,:])):
                if j == 0:
                    fp.writelines([report_matrix[i,j]+', '])
                else:
                    fp.writelines(["%.3f, " % float(report_matrix[i,j])])
            fp.writelines(['\n'])
    fp.close()


    # Create total sum volumes matrix report
    # TODO
    report_sum_vol_name = os.path.join(report_dir, run +'_sum_vol_report.txt')
    np.savetxt(report_sum_vol_name, matrix_sum_volumes, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition matrix report
    report_dep_name = os.path.join(report_dir, run +'_dep_report.txt')
    np.savetxt(report_dep_name, matrix_dep, fmt='%.3f', delimiter=',', newline='\n')

    # Create scour matrix report
    report_sco_name = os.path.join(report_dir, run +'_sco_report.txt')
    np.savetxt(report_sco_name, matrix_sco, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create total active thickness matrix report (calculated from volume matrix)
    report_act_thickness_name = os.path.join(report_dir, run +'_act_thickness_report.txt')
    np.savetxt(report_act_thickness_name, matrix_act_thickness , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition active thickness matrix report (calculated from deposition volume matrix)
    report_act_thickness_name_dep = os.path.join(report_dir, run +'_act_thickness_report_dep.txt')
    np.savetxt(report_act_thickness_name_dep, matrix_act_thickness_dep , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create scour active thickness matrix report (calculated from scour volume matrix)
    report_act_thickness_name_sco = os.path.join(report_dir, run +'_act_thickness_report_sco.txt')
    np.savetxt(report_act_thickness_name_sco, matrix_act_thickness_sco , fmt='%.3f', delimiter=',', newline='\n')
    
    # Create total active area matrix report (calculated from volume matrix)
    report_act_area_name = os.path.join(report_dir, run + '_act_area_report.txt')
    np.savetxt(report_act_area_name, matrix_morph_act_area, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create deposition active area matrix report (calculated from volume matrix)
    report_act_area_name_dep = os.path.join(report_dir, run + '_act_area_report_dep.txt')
    np.savetxt(report_act_area_name_dep, matrix_morph_act_area_dep, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create scour active area matrix report (calculated from volume matrix)
    report_act_area_name_sco = os.path.join(report_dir, run + '_act_area_report_sco.txt')
    np.savetxt(report_act_area_name_sco, matrix_morph_act_area_sco, fmt='%.3f', delimiter=',', newline='\n')

    # Create Wact report matrix
    matrix_Wact=matrix_Wact[:n-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact[:,n-1]=matrix_Wact_Iquantile[:,n-1] # Fill matrix_Wact report with minimum values
    matrix_Wact[:,n]=matrix_Wact_IIIquantile[:,n] # Fill matrix_Wact report with maximum values
    report_Wact_name = os.path.join(report_dir, run +'_morphWact_report.txt')
    np.savetxt(report_Wact_name, matrix_Wact, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create Wact scour report matrix
    matrix_Wact_sco=matrix_Wact_sco[:n-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact_sco[:,n-1]=matrix_Wact_Iquantile_sco[:,n-1] # Fill matrix_Wact report with minimum values
    matrix_Wact_sco[:,n]=matrix_Wact_IIIquantile_sco[:,n] # Fill matrix_Wact report with maximum values
    report_Wact_name_sco = os.path.join(report_dir, run +'_morphWact_sco_report.txt')
    np.savetxt(report_Wact_name_sco, matrix_Wact_sco, fmt='%.3f', delimiter=',', newline='\n')
    
    # Create Wact fill report matrix
    matrix_Wact_dep=matrix_Wact_dep[:n-1,:] # Fill matrix_Wact with morphological  active width values
    matrix_Wact_dep[:,n-1]=matrix_Wact_Iquantile_dep[:,n-1] # Fill matrix_Wact report with minimum values
    matrix_Wact_dep[:,n]=matrix_Wact_IIIquantile_dep[:,n] # Fill matrix_Wact report with maximum values
    report_Wact_name_dep = os.path.join(report_dir, run +'_morphWact_dep_report.txt')
    np.savetxt(report_Wact_name_dep, matrix_Wact_dep, fmt='%.3f', delimiter=',', newline='\n')

    # For each runs collect the dimension of the morphWact_array:
    if delta==1:
        morphWact_dim = np.append(morphWact_dim, len(morphWact_values))


    # Create morphWact/W matrix as following:
    # all morphWact/W values are appended in the same line for each line in the morphWact_values array
    # Now a matrix in which all row are all morphWact/W values for each runs is built
    # morphWact_matrix_header = 'run name, morphWact/W [-]'
    # run name, morphWact/w [-]
    with open(os.path.join(report_dir, run + '_morphWact_array.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values[i])])
        fp.writelines(['\n'])
    fp.close()
    
    with open(os.path.join(report_dir, run + '_morphWact_array_dep.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values_dep[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values_dep[i])])
        fp.writelines(['\n'])
    fp.close()
    
    with open(os.path.join(report_dir, run + '_morphWact_array_sco.txt'), 'w') as fp:
        # fp.write(morphWact_matrix_header)
        # fp.writelines(['\n'])
        for i in range(0, len(morphWact_values)):
            if i == len(morphWact_values)-1:
                fp.writelines(["%.3f" % float(morphWact_values_sco[i])])
            else:
                fp.writelines(["%.3f," % float(morphWact_values_sco[i])])
        fp.writelines(['\n'])
    fp.close()



    # # Print a report with xData as real time in minutes and  the value of scour and deposition volumes for each runs
    # Create report matrix as:
    # run
    # time
    # V_dep
    # V_sco
    
    xData1=np.arange(1, n, 1)*dt_xnr # Time in Txnr
    yData_sco=np.absolute(matrix_sco[:n-1,0])
    yError_sco=matrix_sco[:n-1,-1]
    yData_dep=np.absolute(matrix_dep[:n-1,0])
    yError_dep=matrix_dep[:n-1,-1]
    yData_act_thickness=matrix_act_thickness[:n-1,0]
    yError_act_thickness=matrix_act_thickness[:n-1,-1]
    
    xData2=np.arange(1, n, 1)*dt
    volume_over_time_matrix = []
    volume_over_time_matrix = np.stack((xData2, yData_dep, -yData_sco))

    # Append rows to the current file
    with open(os.path.join(report_dir, 'volume_over_time.txt'), 'a') as fp:
        fp.writelines([run+', '])
        fp.writelines(['\n'])
        for i in range(0,volume_over_time_matrix.shape[0]):
            for j in range(0,volume_over_time_matrix.shape[1]):
                fp.writelines(["%.3f, " % float(volume_over_time_matrix[i,j])])
            fp.writelines(['\n'])
        fp.writelines(['\n'])
    fp.close()
    
    n=0
    for matrix in (DoD_raw_morph_quant, DoD_filt_mean_morph_quant, DoD_filt_isol_morph_quant,DoD_filt_fill_morph_quant,DoD_filt_nature_morph_quant,DoD_filt_isol2_morph_quant,DoD_filt_ult_morph_quant):
        n+=1
        matrix_mean = np.mean(matrix, axis=0)
        matrix_std = np.std(matrix, axis=0)
        if n==1:
            matrix_stack = np.vstack((matrix_mean, matrix_std))
        else:
            matrix_stack = np.vstack((matrix_stack,matrix_mean, matrix_std))
        np.savetxt(os.path.join(report_dir, 'morph_quant_report.txt'), matrix_stack, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        
    # Save report for the analysis of the effects of spatial filter application on DoD, at different stages.
    np.savetxt(os.path.join(report_dir, 'DoD_raw_morph_quant.txt'), DoD_raw_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_mean_morph_quant.txt'), DoD_filt_mean_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_isol_morph_quant.txt'), DoD_filt_isol_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_fill_morph_quant.txt'), DoD_filt_fill_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_nature_morph_quant.txt'), DoD_filt_nature_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_isol2_morph_quant.txt'), DoD_filt_isol2_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    np.savetxt(os.path.join(report_dir, 'DoD_filt_ult_morph_quant.txt'), DoD_filt_ult_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
    
DoD_length_array = np.append(DoD_length_array, DoD_length)


# if run_mode==2:
#     # Print vulume teporal scale report
#     volume_temp_scale_report_header = 'run name, B_dep [min], SD(B_dep) [min], B_sco [min], SD(B_sco) [min]'
#     # Write temporl scale report as:
#     # run name, B_dep, SD(B_dep), B_sco, SD(B_sco)
#     with open(os.path.join(report_dir, 'volume_temp_scale_report.txt'), 'w') as fp:
#         fp.write(volume_temp_scale_report_header)
#         fp.writelines(['\n'])
#         for i in range(0,len(RUNS)):
#             for j in range(0, volume_temp_scale_report.shape[1]+1):
#                 if j == 0:
#                     fp.writelines([RUNS[i]+', '])
#                 else:
#                     fp.writelines(["%.3f, " % float(volume_temp_scale_report[i,j-1])])
#             fp.writelines(['\n'])
#     fp.close()

#%%############################################################################
# STACK ANALYSIS

# for run in RUNS:
#     stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
#     stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
#     # Create stack bool
#     stack_bool=np.where(stack>0, 1, stack)
#     stack_bool=np.where(stack_bool<0, -1, stack_bool)
    
    
#     DoD_length = stack.shape[2] # Define the DoD length L in number of cells
#     DoD_width = stack.shape[1] # Define the DoD width W in number of cells
    
    
#     # COMPUTE THE MORPHOLOGICAL ACTIVE WIDTH EVOLUTIO FOR INCREASING TIMESTEPS
#     envelop_morph_actW_array=[] # This array will contain the morph_actW/W value at increasing delta
#     d=0 # Initialize delta=0
#     for t in range(0, stack.shape[0]-d):
#         # For each available DoD the envelop is calculated dor all the DoD of a given delta
#         envelop=np.nansum(np.abs(stack_bool[:,:,:,d]), axis=0) # Calculate the envelop
#         envelop = np.where(envelop>0,1,envelop)
#         envelop_morph_actW = np.nansum(envelop)/DoD_length/DoD_width # Calculate the morphological active width as morph_actW/W [-]
#         print('t:',t)
#         print('envelop_morph_actW', envelop_morph_actW)
#         envelop_morph_actW_array = np.append(envelop_morph_actW_array, envelop_morph_actW) # Append the morph_actW/W value
#         d+=1
    
    
#     # PLOT
#     plot_dir = os.path.join(home_dir, 'plot', run)
#     fig, axs = plt.subplots(1,1, figsize=(10,6), tight_layout=True)
#     axs.plot(np.linspace(1,stack.shape[0],stack.shape[0]),envelop_morph_actW_array, linestyle='--', marker='^', color='green', label='morph_actW envelop')
#     axs.set_ylim(bottom=0)
#     axs.set_title(run + '- Morphological active width envelop')
#     axs.set_xlabel('DoD timespep')
#     axs.set_ylabel('Morphological active Width/W [-]')
#     axs.legend()
#     plt.savefig(os.path.join(plot_dir, run +'_envelop_morph_act_width.pdf'), dpi=200)
#     plt.show()
    
    
    
#%%############################################################################
end = time.time()
print()
print('Execution time: ', (end-start), 's')
