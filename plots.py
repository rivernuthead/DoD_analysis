#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:15:55 2023

@author: erri
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
data07 = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/DoDs/DoDs_q07_1/DoD_1-0_filt_ult.txt', skiprows=0, delimiter='\t')
data10 = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/DoDs/DoDs_q10_2/DoD_1-0_filt_ult.txt', skiprows=0, delimiter='\t')
data15 = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/DoDs/DoDs_q15_3/DoD_5-4_filt_ult.txt', skiprows=0, delimiter='\t')
data20 = np.loadtxt('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/DoDs/DoDs_q20_2/DoD_2-1_filt_ult.txt', skiprows=0, delimiter='\t')


# Create the figure and axes
fig, ax = plt.subplots()

test = np.where(np.logical_and(data07!=0, np.logical_not(np.isnan(data07))), 1, np.nan)
test1 = np.nansum(np.where(data07!=0, 1, np.nan), axis=0)


# Plot the mean line
ax.plot(np.nansum(np.where(np.logical_and(data07!=0, np.logical_not(np.isnan(data07))), 1, np.nan), axis=0)/np.nansum(np.where(np.logical_not(np.isnan(data07)), 1, np.nan), axis=0), color='#E69F00', label='Q=0.7 l/s')
ax.plot(np.nansum(np.where(np.logical_and(data10!=0, np.logical_not(np.isnan(data10))), 1, np.nan), axis=0)/np.nansum(np.where(np.logical_not(np.isnan(data10)), 1, np.nan), axis=0), color='#009E73', label='Q=1.0 l/s')
ax.plot(np.nansum(np.where(np.logical_and(data15!=0, np.logical_not(np.isnan(data15))), 1, np.nan), axis=0)/np.nansum(np.where(np.logical_not(np.isnan(data15)), 1, np.nan), axis=0), color='#D55E00', label='Q=1.5 l/s')
ax.plot(np.nansum(np.where(np.logical_and(data20!=0, np.logical_not(np.isnan(data20))), 1, np.nan), axis=0)/np.nansum(np.where(np.logical_not(np.isnan(data20)), 1, np.nan), axis=0), color='#CC79A7', label='Q=2.0 l/s')

# ax.fill_between(range(len(np.nanmean(data07, axis=0))), np.nanmin(data07, axis=0), np.nanmax(data07, axis=0), color='#E69F00', alpha=0.3)
# ax.fill_between(range(len(np.nanmean(data10, axis=0))), np.nanmin(data10, axis=0), np.nanmax(data10, axis=0), color='#009E73', alpha=0.3)
# ax.fill_between(range(len(np.nanmean(data15, axis=0))), np.nanmin(data15, axis=0), np.nanmax(data15, axis=0), color='#D55E00', alpha=0.3)
# ax.fill_between(range(len(np.nanmean(data20, axis=0))), np.nanmin(data20, axis=0), np.nanmax(data20, axis=0), color='#CC79A7', alpha=0.3)


# Set labels and title
ax.set_xlabel('Longitudinal coordinate')
ax.set_ylabel('MAW*')
ax.set_title('MAW*')
plt.xticks([])

# Add a legend
ax.legend()

# Save image
plt.savefig('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/' + 'MAW_space_plot.pdf', dpi=1000) # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)
plt.savefig('/home/erri/Documents/PhD/Research/5_research_repos/DoD_analysis/' + 'MAW_space_plot.png', dpi=1000) # raster (png, jpg, rgb, tif), vector (pdf, eps), latex (pgf)