#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:00:11 2021

@author: throop
"""
# This converts from a textfile to markdown
# - Removes crlf connecting adjacent lines, if they are more than 65 characters
# - Leaves unaffected lines which start with -, o, or a number as their first character
# - Escapes $ when not escaped already

file_in = '/Users/throop/notes/house'

lun = open(file_in, "r")

line = ''

with open(file_in) as lun:
    for l in lun:
        
        # Merge this with the previous line, if:
        #  previous line is > 65 characters
        
        # Print the previous line *and* this line, if:
        # 
        
        print(l)
        