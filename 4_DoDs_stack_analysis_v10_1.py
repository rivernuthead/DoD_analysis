#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

Pixel age analysis over stack DoDs

INPUT (as .npy binary files):
    DoD_stack1 : 3D numpy array stack
        Stack on which DoDs are stored as they are, with np.nan
    DoD_stack1_bool : 3D numpy array stack
        Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
OUTPUTS:
    
    
"""
# IMPORT PACKAGES
import os
import numpy as np
from PIL import Image
import math
import random
import time
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
runs = ['q07_1']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

for run in runs:
    # ALENGTH OF THE ANALYSIS WINDOW IN NUMBER OF CELL
    analysis_window = 123 # Number of columns compared to the PiQs photo length
    
    
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    run_dir = os.path.join(home_dir, 'surveys')
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    
    if not(os.path.exists(os.path.join(report_dir, run,  'stack_analysis'))):
        os.mkdir(os.path.join(report_dir, run,  'stack_analysis'))
    
    
    print('###############' +'\n#    ' + run + '\n#' + '##############')
    
    ###############################################################################
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
    
    ###############################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # DATA STRUCTURE
    
    '''
    DoD1-0  DoD2-0  DoD3-0  DoD4-0  DoD5-0  DoD6-0  DoD7-0  DoD8-0  DoD9-0
    DoD2-1  DoD3-1  DoD4-1  DoD5-1  DoD6-1  DoD7-1  DoD8-1  DoD9-1
    DoD3-2  DoD4-2  DoD5-2  DoD6-2  DoD7-2  DoD8-2  DoD9-2
    DoD4-3  DoD5-3  DoD6-3  DoD7-3  DoD8-3  DoD9-3
    DoD5-4  DoD6-4  DoD7-4  DoD8-4  DoD9-4
    DoD6-5  DoD7-5  DoD8-5  DoD9-5
    DoD7-6  DoD8-6  DoD9-6
    DoD8-7  DoD9-7
    DoD9-8
    
    stack = [h,:,:, delta]
    
    '''
    
    
    
    #%%############################################################################
    '''
    DoD envelope
    '''
    # 1 timespan envelope
    for i in range(0,stack.shape[0]):
        envMAW_arr = np.nansum(np.abs(stack_bool[:stack.shape[0]-i,:,:analysis_window ,i]), axis=0) 
        envMAW_arr = np.repeat(envMAW_arr, 51, axis=1)
        envMAW_arr = np.repeat(envMAW_arr, 4, axis=0)
        envMAW = Image.fromarray(np.array(envMAW_arr).astype(np.uint8))
        envMAW.save(os.path.join(report_dir, run, run + '_envMAW_' + str(i+1) + 'tsp.tiff'))
        
        envMAW_bool_arr = np.where(envMAW_arr>0, 1, 0)
        envMAW_bool = Image.fromarray(np.array(envMAW_bool_arr).astype(np.uint8))
        envMAW_bool.save(os.path.join(report_dir, run, run + '_envMAW_bool_' + str(i+1) + 'tsp.tiff'))
        
        


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Define two sample numpy matrices
channel = Image.open('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Photos/q07rgm/Img0001.jpg') # Open image as image
# diff_arr = np.array(diff) # Convert image as numpy array
envMAW = envMAW_arr
envBAW = Image.open('/home/erri/Documents/PhD/Research/5_research_repos/PiQs_analysis/Maps/'+run+'/Img0802_thrs8_cld.tiff')

envMAW = zoom(envMAW, (envBAW.size[1]/envMAW.shape[0], envBAW.size[0]/envMAW.shape[1])) 

# Convert matrices to images
img1 = plt.imshow(channel, alpha=1.0)
img3 = plt.imshow(np.where(envMAW==0, np.nan, envMAW), cmap='cool', aspect=0.1, origin='upper', alpha=0.5)
img2 = plt.imshow(envBAW, cmap='Reds', alpha=0.5, vmin=0, vmax=25)


# Set title and show the plot
plt.title('Overlapping Images with Transparency')
plt.savefig(os.path.join(report_dir, run, run + 'test.pdf'), dpi=600 )
plt.show()
        
        
    
    # #%%############################################################################
    # '''
    # Qual'è la differenza in termini di area tra singoli DoD e DoD inviluppo con delta temporale di 0.5 e 1.0 Txnr?
    # '''
    
    # stack_act = abs(stack_bool) # Create a stack of the activity: 1 means activity, 0 means inactivity
    
    # # Create output matrix as below:
    # #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
    # # delta = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
    # # delta = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
    # # delta = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
    # # delta = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
    # # delta = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
    # # delta = 6  6-0   7-1   8-2   9-3                                average STDEV
    # # delta = 7  7-0   8-1   9-2                                      average STDEV
    # # delta = 8  8-0   9-1                                            average STDEV
    # # delta = 9  9-0                                                  average STDEV
    
    # # stack[time, x, y, delta]
    
    # envelope_DoDs_area_array = []
    # comprehensive_DoD_area_array = []
    
    # t=0
    # d=0
    
    # for t in range(0, stack_act.shape[0]-1):
        
    # # Compute the envelope of the first t+2 DoDs
    #     envelope_DoDs = np.nansum(stack_act[:t+2,:,:,d], axis=0)
    #     envelope_DoDs = np.where(envelope_DoDs>0, 1, np.nan)
        
    #     # Extract the comprehensive DoD of the first t+2 DoDs
    #     comprehensive_DoD = stack_act[0,:,:,t+1]
        
    #     # Compute the active area
    #     envelope_DoDs_area = np.nansum(envelope_DoDs)
    #     comprehensive_DoD_area = np.nansum(comprehensive_DoD)
        
    #     envelope_DoDs_area_array = np.append(envelope_DoDs_area_array, envelope_DoDs_area)
    #     comprehensive_DoD_area_array = np.append(comprehensive_DoD_area_array, comprehensive_DoD_area)
    
    
    # np.savetxt(os.path.join(report_dir, run,  'stack_analysis', run + '_' + 'envelope_DoDs.txt') ,envelope_DoDs_area_array, delimiter=',')
    # np.savetxt(os.path.join(report_dir, run,  'stack_analysis', run + '_' + 'comprehensive_DoD.txt') ,comprehensive_DoD_area_array , delimiter=',')
    
               
    
    
    
    # #%%############################################################################
    # '''
    # Tra un DoD e il successivo quanti pixel nuovi ci sono?
    # Quanti pixel si spengono?
    
    # DoD stack structure.
    # DoD1-0  DoD2-0  DoD3-0  DoD4-0  DoD5-0  DoD6-0  DoD7-0  DoD8-0  DoD9-0
    # DoD2-1  DoD3-1  DoD4-1  DoD5-1  DoD6-1  DoD7-1  DoD8-1  DoD9-1
    # DoD3-2  DoD4-2  DoD5-2  DoD6-2  DoD7-2  DoD8-2  DoD9-2
    # DoD4-3  DoD5-3  DoD6-3  DoD7-3  DoD8-3  DoD9-3
    # DoD5-4  DoD6-4  DoD7-4  DoD8-4  DoD9-4
    # DoD6-5  DoD7-5  DoD8-5  DoD9-5
    # DoD7-6  DoD8-6  DoD9-6
    # DoD8-7  DoD9-7
    # DoD9-8
    
    # '''
    # for delta in [0,1,2]:
    #     for t in range(0,stack_bool.shape[0]):
    #         DoDs_diff = abs(stack_bool[(delta+1)*(t+1), :,:,delta]) - abs(stack_bool[(delta+1)*t,:,:,delta])
    #         report_OnOffmatrix = np.zeros((stack_bool.shape[0]-1, 2))
    #         for t in range(0,report_OnOffmatrix.shape[0]):
    #             positive = np.nansum(DoDs_diff[t,:,:]>0)
    #             negative = np.nansum(DoDs_diff[t,:,:]<0)
    #             report_OnOffmatrix[t,0] = positive
    #             report_OnOffmatrix[t,1] = -negative
    
    #     header = 'Pixel On, Pixel Off'
    #     np.savetxt(os.path.join(report_dir, run,  'stack_analysis', run + '_' + 'delta'+ str(delta) + '_' + 'OnOff_pixel.txt') ,report_OnOffmatrix, delimiter=',', header=header)
    
    
    # #%%############################################################################
    # end = time.time()
    # print()
    # print('Execution time: ', (end-start), 's')