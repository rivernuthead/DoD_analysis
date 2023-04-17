#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:29:14 2023

@author: erri
"""

import matplotlib.pyplot as plt
import PyPDF2

# Generate matplotlib chart
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('My chart')

# Save chart as PDF
plt.savefig('my_chart.pdf', bbox_inches='tight')
plt.show()

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('My chart 2')

# Save chart as PDF
plt.savefig('my_chart2.pdf', bbox_inches='tight')
plt.show()


merger = PyPDF2.PdfFileMerger()

# Open and append the existing PDF
with open("my_chart.pdf", "rb") as existing_file:
    merger.append(existing_file)

# Open and append the new PDF chart
with open("my_chart2.pdf", "rb") as chart_file:
    merger.append(chart_file)

# Save the merged PDF
with open("merged_file.pdf", "wb") as merged_file:
    merger.write(merged_file)