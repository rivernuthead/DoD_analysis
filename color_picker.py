#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:45:44 2024

@author: erri
"""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# Get the Viridis color map
viridis_cmap = plt.get_cmap('coolwarm_r')

# Number of colors you want to pick
num_colors = 7

# Get equally spaced colors from the Viridis color map
equally_spaced_colors = [to_hex(viridis_cmap(i / (num_colors - 1))) for i in range(num_colors)]

# Display the result
print(equally_spaced_colors)
