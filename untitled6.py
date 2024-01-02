#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:40:54 2023

@author: erri
"""

import numpy as np

def keep_every_j_elements(arr, j):
    if j <= 0:
        raise ValueError("J should be a positive integer")

    result = arr[::j]
    return result

# Example usage:
original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
J = 2  # Replace this with your desired value of J
result_array = keep_every_j_elements(original_array, J)

print("Original array:", original_array)
print(f"Keep every {J}th element:", result_array)

