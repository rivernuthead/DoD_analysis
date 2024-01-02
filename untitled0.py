#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:07:00 2023

@author: erri
"""

import numpy as np

# Create a numpy stack containing real numbers (you can replace this with your actual data)
# Here's an example with 1000 random real numbers between 0 and 100
numpy_stack = np.random.rand(1000) * 100

import numpy as np
import matplotlib.pyplot as plt

# Assuming you already have your numpy stack containing real numbers

# Compute the frequency distribution
hist, bin_edges = np.histogram(numpy_stack, bins='auto')

# Plot the histogram
plt.hist(numpy_stack, bins='auto', color='blue', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Real Numbers')
plt.grid(True)
plt.show()
