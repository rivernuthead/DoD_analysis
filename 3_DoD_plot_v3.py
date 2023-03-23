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
from DoD_analysis_functions import *
from DoD_analysis_functions_3 import *
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
    1 = one run at time
    2 = bath process
DoD_map_plot:
    0 = plot DoD map OFF
    1 = plot DoD map ON
'''
run_mode = 2
DoD_map_plot = 0

# SINGLE RUN NAME
run = ['q15_3']

DoD_name = 'DoD_1-0'

# # Set DEM single name to perform process to specific DEM
# DEM1_single_name = 'matrix_bed_norm_' + run +'s'+'0'+'.txt' # DEM1 name
# DEM2_single_name = 'matrix_bed_norm_' + run +'s'+'1'+'.txt' # DEM2 name

# # Filtering process thresholds values
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


###############################################################################
# SETUP FOLDERS and RUNS
###############################################################################
# setup working directory and DEM's name
home_dir = os.getcwd()
DoDs_dir = os.path.join(home_dir, 'DoDs')
report_dir = os.path.join(home_dir, 'output')
run_dir = os.path.join(home_dir, 'surveys')
main_plot_dir = os.path.join(home_dir, 'plot')



# Create the run name list
RUNS=[]
if run_mode ==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run # RUNS as a single entry array, provided by run variable


###############################################################################
# INITIALIZE ARRAYS




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
    input_dir = os.path.join(home_dir, 'surveys', run)
    report_path = os.path.join(report_dir, run)
    plot_dir = os.path.join(home_dir, 'plot', run)
    DoDs_dir = os.path.join(home_dir, 'DoDs', 'DoD_'+run)
    
    # Save a report with xData as real time in minutes and the value of scour and deposition volumes for each runs
    # Check if the file already exists
    if os.path.exists(os.path.join(report_path, 'volume_over_time.txt')):
        os.remove(os.path.join(report_path, 'volume_over_time.txt'))
    else:
        pass

    # CREATE FOLDERS
    if not(os.path.exists(report_path)):
        os.mkdir(report_path)
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

    # # Run discharge
    Q = run_param[0,0] # Run discharge [l/s]
    
    # Create the list of surveys files and sore them in the files list.
    # To do so, two system array will be created for the run with more than 10 surveys due to the behaviour of the sort python function that sort files in this way:
    # matrix_bed_norm_q10_3s0.txt matrix_bed_norm_q10_3s1.txt matrix_bed_norm_q10_3s11.txt matrix_bed_norm_q10_3s12.txt matrix_bed_norm_q10_3s2.txt
    files = []
    files1=[] # Files list with surveys from 0 to 9
    files2=[] # Files list with surveys from 10
    # Creating array with file names:
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'s'):
            files = np.append(files, f)
    for f in files:
        if len(f)==len(files[1]):
            files1 = np.append(files1, f)
        elif len(f)==len(files[1])+1:
            files2 = np.append(files2, f)
            
    files = np.append(files1,files2) # Files has been overwritten with a list of file names sorted in the right way :) 
    
    if DoD_map_plot == 1:
        # PLOT ONE DoD AT THE FINAL FILTERING STAGE
        DoD_filt_ult = np.loadtxt(os.path.join(DoDs_dir, DoD_name+'_filt_ult.txt'))
        fig, ax = plt.subplots(dpi=200, tight_layout=True)
        # im = ax.imshow(np.where(DoD_filt_isol2_gis==NaN, np.nan, DoD_filt_ult_gis), cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
        im = ax.imshow(DoD_filt_ult, cmap='RdBu',  vmin=-25, vmax=25, aspect='0.1')
        # plt.colorbar(im)
        plt.title(DoD_name + '-' + run, fontweight='bold')
        plt.savefig(os.path.join(plot_dir, run + '_' + DoD_name+'.pdf'), dpi=1600)
        plt.show()
                
        # PLOT OF ALL THE DIFFERENT FILTERING STAGE
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1, tight_layout=True, figsize=(10,6))
        fig.suptitle('Filtering process - ' + run)
        
        # Import files:
        DoD_raw = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_raw.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_mean = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_mean.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_isol = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_isol.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_nature = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_nature.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_fill = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_fill.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_isol2 = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_isol2.txt'),skiprows=8 , delimiter='\t')
        DoD_filt_ult = np.loadtxt(os.path.join(DoDs_dir, DoD_name + '_filt_ult.txt'),skiprows=8 , delimiter='\t')
        
        # Convert all zero value in np.nan to make it transparent on plots:
        DoD_raw_plot = np.where(DoD_raw==0, np.nan, DoD_raw)
        DoD_filt_mean_plot = np.array(np.where(DoD_filt_mean==0, np.nan, DoD_filt_mean))
        DoD_filt_isol_plot = np.array(np.where(DoD_filt_isol==0, np.nan, DoD_filt_isol))
        DoD_filt_nature_plot = np.array(np.where(DoD_filt_nature==0, np.nan, DoD_filt_nature))
        DoD_filt_fill_plot = np.array(np.where(DoD_filt_fill==0, np.nan, DoD_filt_fill))
        DoD_filt_isol2_plot = np.array(np.where(DoD_filt_isol2==0, np.nan, DoD_filt_isol2))
        DoD_filt_ult_plot = np.array(np.where(DoD_filt_ult==0, np.nan, DoD_filt_ult))
        
        raw = ax1.imshow(DoD_raw_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax1.set_title('raw DoD')
    
        filt_mean = ax2.imshow(DoD_filt_mean_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax2.set_title('filt_mean')
    
        filt_isol = ax3.imshow(DoD_filt_isol_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax3.set_title('filt_isol')
        
        filt_nature = ax4.imshow(DoD_filt_nature_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax4.set_title('filt_nature')
        
        filt_fill = ax5.imshow(DoD_filt_fill_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax5.set_title('filt_fill')
        
        filt_isol2 = ax6.imshow(DoD_filt_isol2_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax6.set_title('filt_isol2')
        
        filt_ult = ax7.imshow(DoD_filt_ult_plot, cmap='RdBu', vmin=-50, vmax=50, aspect='0.1')
        ax7.set_title('filt_ult')
        
        # fig.colorbar(DoD_filt_isol2_plot)
        plt.savefig(os.path.join(plot_dir,'filtmap_'+run +'_'+DoD_name+'_filtmap.pdf'), dpi=1000) # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)
        plt.show()

    


    ###########################################################################
    # PLOTS
    ###########################################################################
    #Load data:
    matrix_sco = np.loadtxt(os.path.join(report_path, run+'_sco_report.txt'), delimiter=',')
    matrix_dep = np.loadtxt(os.path.join(report_path, run+'_dep_report.txt'), delimiter=',')
    matrix_act_thickness = np.loadtxt(os.path.join(report_path, run+'_act_thickness_report.txt'), delimiter=',')
    matrix_act_thickness_dep = np.loadtxt(os.path.join(report_path, run+'_act_thickness_report_dep.txt'), delimiter=',')
    matrix_act_thickness_sco = np.loadtxt(os.path.join(report_path, run+'_act_thickness_report_sco.txt'), delimiter=',')
    matrix_morphWact = np.loadtxt(os.path.join(report_path, run+'_morphWact_report.txt'), delimiter=',')
    matrix_morphWact_sco = np.loadtxt(os.path.join(report_path, run+'_morphWact_sco_report.txt'), delimiter=',')
    matrix_morphWact_dep = np.loadtxt(os.path.join(report_path, run+'_morphWact_dep_report.txt'), delimiter=',')
    
    
    # Define arrays for scour and volume data over time
    xData1=np.arange(1, len(files), 1)*dt_xnr # Time in Txnr
    
    yData_sco=np.absolute(matrix_sco[:len(files)-1,-2])
    yError_sco=matrix_sco[:len(files)-1,-1]
    
    yData_dep=np.absolute(matrix_dep[:len(files)-1,-2])
    yError_dep=matrix_dep[:len(files)-1,-1]
    
    yData_act_thickness=matrix_act_thickness[:len(files)-1,-2]
    yError_act_thickness=matrix_act_thickness[:len(files)-1,-1]
    
    yData_act_thickness_dep=matrix_act_thickness_dep[:len(files)-1,-2]
    yError_act_thickness_dep=matrix_act_thickness_dep[:len(files)-1,-1]
    
    yData_act_thickness_sco=matrix_act_thickness_sco[:len(files)-1,-2]
    yError_act_thickness_sco=matrix_act_thickness_sco[:len(files)-1,-1]
    
    yData_morphWact=matrix_morphWact[:len(files)-1,-2]
    yError_morphWact=matrix_morphWact[:len(files)-1,-1]
    
    yData_morphWact_sco=matrix_morphWact_sco[:len(files)-1,-2]
    yError_morphWact_sco=matrix_morphWact_sco[:len(files)-1,-1]
    
    yData_morphWact_dep=matrix_morphWact_dep[:len(files)-1,-2]
    yError_morphWact_dep=matrix_morphWact_dep[:len(files)-1,-1]
    

    fig3, axs = plt.subplots(2,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
    fig3.suptitle(run + ' - Volume')
    axs[0].errorbar(xData1,yData_sco, yError_sco, linestyle='--', marker='^', color='red')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Scour')
    # axs[0].set_xlabel()
    axs[0].set_ylabel('Scour volume [unit]')
    axs[1].errorbar(xData1,yData_dep, yError_dep, linestyle='--', marker='^', color='blue')
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Deposition')
    axs[1].set_xlabel('Exner time')
    axs[1].set_ylabel('Deposition volume [unit]')
    plt.savefig(os.path.join(plot_dir, run +'dep_scour.pdf'), dpi=200)
    plt.show()
    
    
    fig4, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
    axs.errorbar(xData1,yData_act_thickness, yError_act_thickness, linestyle='--', marker='^', color='purple')
    axs.set_ylim(bottom=0)
    axs.set_title(run + '- Morphological active layer')
    axs.set_xlabel('Exner time')
    axs.set_ylabel('Active thickness [mm]')
    plt.savefig(os.path.join(plot_dir, run +'_morph_active_layer.pdf'), dpi=200)
    plt.show()
    
    fig5, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
    axs.errorbar(xData1,yData_morphWact, yError_morphWact, linestyle='--', marker='^', color='green')
    axs.set_ylim(bottom=0)
    axs.set_title(run + '- Morphological active width')
    axs.set_xlabel('Exner time')
    axs.set_ylabel('Morphological active Width / W [-]')
    plt.savefig(os.path.join(plot_dir, run +'_morph_act_width.pdf'), dpi=200)
    plt.show()
    
    fig5, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
    axs.errorbar(xData1,yData_morphWact_sco, yError_morphWact_sco, linestyle='--', marker='^', color='red', label='morphWact sco')
    axs.errorbar(xData1,yData_morphWact_dep, yError_morphWact_dep, linestyle='--', marker='^', color='blue', label='morphWact dep')
    axs.set_ylim(bottom=0)
    axs.set_title(run + '- Morphological active width')
    axs.set_xlabel('Exner time')
    axs.set_ylabel('Morphological active Width / W [-]')
    axs.legend()
    plt.savefig(os.path.join(plot_dir, run +'_morph_act_width_sco_dep.pdf'), dpi=200)
    plt.show()
    
    fig6, axs = plt.subplots(1,1,dpi=80, figsize=(10,6), sharex=True, tight_layout=True)
    axs.errorbar(xData1,yData_act_thickness_sco, yError_act_thickness_sco, linestyle='--', marker='^', color='red', label='morph act layer sco')
    axs.errorbar(xData1,yData_act_thickness_dep, yError_act_thickness_dep, linestyle='--', marker='^', color='blue', label='morph act layer dep')
    axs.set_ylim(bottom=0)
    axs.set_title(run + '- Morphological active layer')
    axs.set_xlabel('Exner time')
    axs.set_ylabel('Morphological active layer [mm]')
    axs.legend()
    plt.savefig(os.path.join(plot_dir, run +'_morph_act_layer_sco_dep.pdf'), dpi=200)
    plt.show()
    
    # Multiple boxplot of the morphological active layer for different timestep
    fig7, ax = plt.subplots(dpi=80, figsize=(10,6))
    fig7.suptitle('Morphological active layer - ' + run, fontsize = 18)
    for i in range(0, matrix_act_thickness.shape[0]):
        bplot=ax.boxplot(matrix_act_thickness[i,:-2-i], positions=[i], widths=0.5) # Data were filtered by np.nan values
    ax.yaxis.grid(True)
    ax.set_xlabel('DoD timestep', fontsize=12)
    ax.set_ylabel('Morphological active layer [mm]', fontsize=12)
    # plt.xticks(np.arange(0,matrix_act_thickness.shape[0], 1), RUNS)
    # if run_mode==2:
    #     plt.savefig(os.path.join(home_dir, 'plot', 'morph_act_width_boxplot.pdf'), dpi=200)
    plt.savefig(os.path.join(plot_dir, 'morph_act_layer_boxplot.pdf'), dpi=200)
    plt.show()
    
###############################################################################
# MORPHOLOGICAL ACTIVE WIDTH VS. DISCHARGE BOXPLOT
morphWact_dim=[]
morphWact_dim_dep=[]
morphWact_dim_sco=[]

# Check the maximum matrix dimension
for i in range(0,len(RUNS)):
    report_dir_data = os.path.join(home_dir, 'output', RUNS[i])
    morphWact = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array.txt'), delimiter=',') # Morphological active width
    morphWact_dep = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array_dep.txt'), delimiter=',') # Morphological active width DEPOSITION
    morphWact_sco = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array_sco.txt'), delimiter=',') # Morphological active width SCOUR
    morphWact_dim =np.append(morphWact_dim, len(morphWact))
    morphWact_dim_dep =np.append(morphWact_dim, len(morphWact_dep))
    morphWact_dim_sco =np.append(morphWact_dim, len(morphWact_sco))

NaN_value = -999

# Initialize report matrix
# The matrix has the number of colums as the maximum number of columns of the dataset.
# So I initilize it as filled with the NaN_value and then I convert NaN_value as np.nan.
# This allows me to consider the dataset with their right dimension in number of columns.
morphWact_matrix=np.ones((len(RUNS), int(np.max(morphWact_dim))))*NaN_value
morphWact_matrix_dep=np.ones((len(RUNS), int(np.max(morphWact_dim_dep))))*NaN_value
morphWact_matrix_sco=np.ones((len(RUNS), int(np.max(morphWact_dim_sco))))*NaN_value


for i in range(0,len(RUNS)):
    report_dir_data = os.path.join(home_dir, 'output', RUNS[i])
    morphWact = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array.txt'), delimiter=',')
    morphWact_dep = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array_dep.txt'), delimiter=',') # Morphological active width DEPOSITION
    morphWact_sco = np.loadtxt(os.path.join(report_dir_data, RUNS[i] + '_morphWact_array_sco.txt'), delimiter=',') # Morphological active width SCOUR
    # morphWact_dim =len(morphWact)
    morphWact_matrix[i,:len(morphWact)]=morphWact
    morphWact_matrix_dep[i,:len(morphWact_dep)]=morphWact_dep
    morphWact_matrix_sco[i,:len(morphWact_sco)]=morphWact_sco

# Set NaN_value as np.nan
morphWact_matrix = np.where(morphWact_matrix==NaN_value, np.nan, morphWact_matrix)
morphWact_matrix_dep = np.where(morphWact_matrix_dep==NaN_value, np.nan, morphWact_matrix_dep)
morphWact_matrix_sco = np.where(morphWact_matrix_sco==NaN_value, np.nan, morphWact_matrix_sco)


# MULTIPLE BOXPLOT OF THE MORPHOLOGICAL ACTIVE WIDTH DATA
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Dimensionless morphological active width', fontsize = 18)
for i in range(0, len(RUNS)):
    bplot=ax.boxplot(morphWact_matrix[i,:][~np.isnan(morphWact_matrix[i,:])], positions=[i], widths=0.5) # Data were filtered by np.nan values
ax.yaxis.grid(True)
ax.set_xlabel('Runs', fontsize=12)
ax.set_ylabel('morphWact/W [-]', fontsize=12)
plt.xticks(np.arange(0,len(RUNS), 1), RUNS)
if run_mode==2:
    plt.savefig(os.path.join(home_dir, 'plot', 'morph_act_width_boxplot.pdf'), dpi=200)
plt.savefig(os.path.join(main_plot_dir, 'morph_act_width_boxplot.pdf'), dpi=200)
plt.show()


# MULTIPLE BOXPLOT OF THE MORPHOLOGICAL ACTIVE WIDTH DATA FOR SCOUR AND DEPOSITION
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Dimensionless dep and scour morphological active width', fontsize = 18)
positions = np.linspace(0,len(RUNS)*2, len(RUNS)+1)
for i in range(0,len(RUNS)):
    bplot_dep=ax.boxplot(morphWact_matrix_dep[i,:][~np.isnan(morphWact_matrix_dep[i,:])], positions=[int(positions[i])], widths=0.5, patch_artist=True, boxprops=dict(facecolor='b')) # Data were filtered by np.nan values
    bplot_sco=ax.boxplot(morphWact_matrix_sco[i,:][~np.isnan(morphWact_matrix_sco[i,:])], positions=[int(positions[i])+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='r'))
ax.yaxis.grid(True)
ax.set_xlabel('Runs', fontsize=12)
ax.set_ylabel('morphWact/W [-]', fontsize=12)
plt.xticks(np.arange(0,len(RUNS)*2, 2), RUNS)
if run_mode==2:
    plt.savefig(os.path.join(home_dir, 'plot', 'morph_act_width_boxplot_dep_sco.pdf'), dpi=200)
plt.savefig(os.path.join(main_plot_dir, 'morph_act_width_boxplot_dep_sco.pdf'), dpi=200)
plt.show()


#%%############################################################################
# STACK ANALYSIS PLOT
# TODO Move this part in the stack analysis script!
envelop_intersection_1_max_timestep_array=[]

for run in RUNS:
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    # Create stack bool
    stack_bool=np.where(stack>0, 1, stack)
    stack_bool=np.where(stack_bool<0, -1, stack_bool)
    
    
    DoD_length = stack.shape[2] # Define the DoD length L in number of cells
    DoD_width = stack.shape[1] # Define the DoD width W in number of cells
    
    ###########################################################################
    # COMPUTE THE MORPHOLOGICAL ACTIVE WIDTH EVOLUTIO FOR INCREASING TIMESTEPS
    envelop_morph_actW_array=[] # This array will contain the morph_actW/W value at increasing delta
    d=0 # Initialize delta=0
    for t in range(0, stack.shape[0]-d):
        # For each available DoD the envelop is calculated dor all the DoD of a given delta
        envelop=np.nansum(np.abs(stack_bool[:,:,:,d]), axis=0) # Calculate the envelop
        envelop = np.where(envelop>0,1,envelop)
        envelop_morph_actW = np.nansum(envelop)/DoD_length/DoD_width # Calculate the morphological active width as morph_actW/W [-]
        envelop_morph_actW_array = np.append(envelop_morph_actW_array, envelop_morph_actW) # Append the morph_actW/W value
        d+=1
    
    
    # PLOT THE MORPHOLOGICAL ACTIVE WIDTH OF THE ENVELOP EVOLUTION
    plot_dir = os.path.join(home_dir, 'plot', run)
    fig, axs = plt.subplots(1,1, figsize=(10,6), tight_layout=True)
    axs.plot(np.linspace(1,stack.shape[0],stack.shape[0]),envelop_morph_actW_array, linestyle='--', marker='^', color='green', label='morph_actW envelop')
    axs.set_ylim(bottom=0)
    axs.set_title(run + '- Morphological active width envelop')
    axs.set_xlabel('DoD timespep')
    axs.set_ylabel('Morphological active Width/W [-]')
    axs.legend()
    plt.savefig(os.path.join(plot_dir, run +'_envelop_morph_act_width.pdf'), dpi=200)
    plt.show()
    
    
    ###########################################################################
    # COMPUTE THE DIFFERENCE BETWEEN THE ENVELOP OF THE ALL 1-TIMESTEP DoD AND
    # THE MAXIMUM TIMESTEP DoD
    envelop_1timestep = np.nansum(np.abs(stack_bool[:,:,:,0]), axis=0) # Calculate the envelop of all the 1-timestep DoD
    envelop_1timestep = np.where(envelop_1timestep>0, 1, envelop_1timestep)
    envelop_max_timestep = np.abs(stack_bool[0,:,:,-1]) # Extract the maximum timestep DoD
    
    envelop_1timestep = np.where(envelop_1timestep==0, np.nan, envelop_1timestep)
    envelop_max_timestep = np.where(envelop_max_timestep==0, np.nan, envelop_max_timestep)
    envelop_intersection_1_max_timestep_map = envelop_1timestep*envelop_max_timestep
    envelop_intersection_1_max_timestep_array=np.append(envelop_intersection_1_max_timestep_array, np.nansum(envelop_intersection_1_max_timestep_map)/np.nansum(envelop_1timestep))
    
    fig, ax = plt.subplots(dpi=200, tight_layout=True)
    im1 = ax.imshow(envelop_1timestep, cmap='summer', aspect='0.1', alpha=0.8) # ,  vmin=-25, vmax=25
    im2 = ax.imshow(envelop_max_timestep, cmap='spring', aspect='0.1', alpha=0.7) # ,  vmin=-25, vmax=25
    plt.title('Envelop'+ '-' + run)
    plt.savefig(os.path.join(plot_dir, 'envelope_' + run +'.pdf'), dpi=1600)
    plt.show()

    
#%%
end = time.time()
print()
print('Execution time: ', (end-start), 's')
