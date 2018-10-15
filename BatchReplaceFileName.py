# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:08:22 2018

@author: A-Bibeka
"""

import os

print('Current working directory ',os.getcwd())
os.chdir('C:/Users/a-bibeka/Dropbox/TTI_Projects/Road User Cost/VISSIM AM Peak V10/NB/NB 2025') 
print('Current working directory ',os.getcwd())
# Replace _ to - in a file name 
#[os.rename(f, f.replace('_', '-')) for f in os.listdir('.') if not f.startswith('.')]
# Replace the year from 2045 to 2025
[os.rename(f, f.replace('2045', '2025')) for f in os.listdir('.') if not f.startswith('.')]