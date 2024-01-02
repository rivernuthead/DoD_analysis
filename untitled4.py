#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:10:32 2023

@author: erri
"""

import numpy as np

def check_conditions(array1, array2):
    condition1 = np.any((array1 == 1) & (array2 == 0))
    condition2 = np.any(array1 == -1) & np.any(array1 == 1) & np.all(array2 == 0)
    condition3 = np.all((array1 == 0) & (array2 == 1))
    
    # condition1 = np.any((array1 == 1)) and (array2 == 0)
    # condition2 = np.any((array1 == -1) & (array1 == 1)) and (array2 == 0)
    # condition3 = np.all((array1 == 0)) and (array2 == 1)

    return condition1, condition2, condition3


    
array1 = np.array([0, 0, 0])
array2 = np.array([1])

result = check_conditions(array1, array2)

print("Condition 1:", result[0]*1)
print("Condition 2:", result[1]*1)
print("Condition 3:", result[2]*1)
