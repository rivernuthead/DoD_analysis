#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:21:09 2022

@author: erri
"""


import time
import os
import subprocess

start = time.time() # Set initial time
script = 'DoD_analysis_v12.py'
home_dir = os.getcwd()
script_path = os.path.join(home_dir, script)
run_dir = os.path.join(home_dir, 'surveys')

for RUN in sorted(os.listdir(run_dir)):
    if RUN.startswith('q'):
        print()
        print(RUN)
        print()
        # exec(open(os.path.join(home_dir,"DoD_analysis_v1.2.py")).read())
        subprocess.call(script_path, shell=True)


end = time.time()