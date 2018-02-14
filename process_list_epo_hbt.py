#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:30:17 2018

This file is to process my list of EPO talks etc into a format for Mark Sykes' Word document.

1. Open file 'HBT Outreach EPO List.numbers'
2. Cut & Paste from there, into txt file in vi.
3. Run this script on that file
4. Cut & Paste into Word
5. Add some <cr> at the end of each line (not sure why)

@author: throop
"""


f = open('outreach_hbt_2017', 'r')
text = f.read()
f.close()
lines = text.split('\n')

lines = lines[-1:0:-1]

for line in lines:
    if (line):
        (date, city, country, venue, title) = line.split('\t')
        title = title.replace('New Horizons', "NASA's New Horizons Mission to Pluto")
        title = title.replace("Astrobiology", "Astrobiology: Are We Alone?")
        title = title.replace("Star Formation", "Planet Formation in Dense Star Clusters")
        line_out = f'{date}, {city}, {country} (HBT) -- {venue}, "{title}"\n'
        print(line_out)
print(f"\nPrinted {len(lines)} events.")    
