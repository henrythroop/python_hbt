#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:22:10 2020

@author: throop
"""

import numpy as np
import xlrd as xlrd

#import hbtshort
import re
import glob
import os

def abbreviate(s):

    a = [
            
## Abbreviate institution names
            
    ('Johns Hopkins University/Applied Physics Laboratory', 'JHU-APL'),
    ('SETI Institute', 'SETI'),
    ('Jet Propulsion Laboratory', 'JPL'),
    ('NASA Ames Research Center', 'NASA Ames'),
    ('Lowell Observatory', 'Lowell'),
    ('Johns Hopkins University', 'JHU'),
    ('University of Maryland, College Park', 'UMD'),
    ('Brown University', 'Brown U'),
    ('NASA Johnson Space Center', 'NASA JSC'),
    ('University Of Colorado, Boulder', 'U Colorado'),
    ('Southwest Research Institute', 'SwRI'),
    ('Space Science Institute', 'SSI'),
    ('Arizona State University', 'ASU'),
    ('University of New Mexico', 'UNM'),
    ('University Of California, Santa Cruz', 'UCSC'),
    ('NASA Foreign PI Support Organization', 'Foreign'),
    ('Florida Institute Of Technology', 'FIT'),
    ('State University Of New York, ', 'SUNY '),
    ('Lawrence Livermore National Security, LLC', 'LLNL'),
    ('University Of Idaho, Moscow', 'U Idaho'),
    ('University Of Arizona', 'U Arizona'),
    ('University of California, Los Angeles', 'UCLA'),
    ('University of Hawaii, Honolulu', 'U Hawaii'),
    ('Purdue University', 'Purdue'),
    ('University of Western Ontario', 'U Western Ontario'),
    ('Planetary Science Institute', 'PSI'),
    ('Auburn University', 'Auburn'),
    ('Catholic University of America', 'Catholic'),
    ('University of Central Florida', 'UCF'),
    ('NASA Goddard Space Flight Center', 'NASA GSFC'),
    ('US Naval Academy', 'Naval Acad'),
    ('University of California, Berkeley', 'Berkeley'),
    ('University Of Alaska, Fairbanks', 'U Alaska'),
    ('California Institute of Technology', 'Caltech'),
    ('Princeton University', 'Princeton'),
    ('Cornell University', 'Cornell'),
    ('Trinity University', 'Trinity'),
    ('Naval Research Lab', 'NRL'),
    ('Imperial College Of Science Technology & Medicine', 'Imperial College'),
    ('Brigham Young University', 'BYU'),
    ('Massachusetts Institute of Technology', 'MIT'),
    ('Northern Arizona University', 'NAU'),
    ('NASA Marshall Space Flight Center', 'NASA MSFC'),
    ('Institute For Advanced Study', 'Princeton-IAS'),
    ('CTRE NAT DE LA RECHERCHE SCIENTIFIQUE', 'CNRS France'),
    ('Centre National de la Recherche Scientifique', 'CNRS France'),
    ('Embry-Riddle Aeronautical University, Inc.', 'Embry-Riddle'),
    ('University of California, ', 'UC '),
    ('University of Michigan', 'U Mich'),
    ('Georgia Tech Research', 'GA Tech'),
    ('North Carolina State', 'NC State'),
    ('THE REGENTS OF THE UNIVERSITY OF CALIFORNIA', 'UC'),
    ('Smithsonian Institution', 'Smithsonian'),
    ('University of Texas, El Paso', 'UTEP'),
    ('University of Tennessee, Knoxville', 'UTK'),
    ('Universities Space Research Association, Columbia', 'LPI'),
    ('Rutgers University', 'Rutgers'),
    ('University of Texas, San Antonio', 'UTSA'),
    ('University of New Hampshire', 'UNH'),
    ('University of Arkansas, Fayetteville', 'U Ark'),
    ('Hampton University', 'Hampton'),
    ('President and Fellows of Harvard College', 'Harvard'),
    ('American University', 'American U'),
    ('Lockheed Martin Inc.', 'Lockheed'),
    ('University of Wisconsin, Madison', 'U Wisc'),
    ('Lawrence Berkeley National Laboratory', 'Lawrence Berkeley NL'),
    ('Los Alamos National Security', 'LANL'),
    ('Dartmouth College', 'Dartmouth'),
    ('University Of Maryland Baltimore County', 'UMD, Baltimore'),
    ('University Potsdam, Institute of physics and astronomy, Germany', 'U Potsdam, Germany'),
    ('New Mexico Institute Of Mining And Technology', 'NM Tech'),
    ('Spece Environmen Technologies', 'Space Env Tech'),
    ('LESIA, Paris Observatory, France', 'LESIA, France'),
    ('ISTITUTO NAZIONALE DI ASTROFISICA INAF', 'INAF Italy'),
    ('University of Virginia, Charlottesville', 'UVA'),
    ('The Pinhead', 'Pinhead'),

# Change some styles
    
    ('SELF', 'Self'),
    ('OXFORD', 'Oxford'),

# Remove some campus names, for the main campus
    
    (', THE (INC)', ''),
    (', Iowa City', ''),
    (', Ann Arbor', ''),
    (', Austin', ''),
    (', Lafayette', ''),
    (', Athens', ''),
    (', Durham', ''),
    (', New Brunswick', ''),
    (' DR14', ''),
    (' Flagstaff', ''),
    (' and A&M College', ''),
    

## Abbreviate a few common phrases
    
    ('Corporation', ''),
    ('State University', 'State'),
    (', LLC', ''),
    ('University Of ', 'U '),
    ('University', 'U'),
    ('Universitaet', 'U'),
    ('National Laboratory', 'NL'),
    ('Research Center', ''),
    (', Inc.', ''),
    
## Abbreviate the roles
    
    ('Collaborator', 'Collab'),
    ('Science PI', 'Sci-PI'),
    ('Graduate/Undergraduate Student', 'Student'),
    ('(non-US organization only)', 'FOREIGN'),
    ('Postdoctoral Associate', 'Postdoc'),
    ]

    for t in a:
        insensitive = re.compile(t[0], re.IGNORECASE)
        s = insensitive.sub(t[1], s)
        # s = s.replace(t[0], t[1])
        
    return s

file_xl = '/Users/hthroop/Downloads/SSW_Volcanism.xls'
file_xl = '/Users/hthroop/Downloads/CDAP20_ATM.xls'

files_xl = glob.glob('/Users/hthroop/Documents/HQ/CDAP20/MaRIE/Panel_Compilation*xls')

DO_LIST_FOR_GOOGLE = True
DO_LIST_FOR_NICY = not(DO_LIST_FOR_GOOGLE)

for file_xl in files_xl:

    name_panel = os.path.basename(file_xl).replace('Panel_Compilation_','').replace('.xls', '')
    
    print('------------------------')
    print(f'{name_panel}')
    print('------------------------')
    
    workbook = xlrd.open_workbook(file_xl)
    sheet_names = workbook.sheet_names()
    # print('Sheet Names', sheet_names)
    
    num_proposals = len(sheet_names)-3
    #sheet_proposals = workbook.sheet_proposals
    
    institutions = np.zeros(num_proposals).astype(set)
    
    for i in range(num_proposals):
        
        # institutions[i] = {}  # A set, which is unordered, and does not allow duplicates
        
        sheet = workbook.sheet_by_index(i+3)
        num_investigators = sheet.nrows-1

        num_proposal = sheet_names[i+3]

        if DO_LIST_FOR_GOOGLE:                    
            print(num_proposal)
            print('------')
        
        institutions[i] = {}  # blank set
        
        for j in range(num_investigators):
            
            role = sheet.cell_value(j+1,0)  # row (1st row is 0), column (1st col is A)
            role = abbreviate(role)
            
            name_last = sheet.cell_value(j+1,3)
            name_first = sheet.cell_value(j+1,4)
            
            if (role == 'PI'):
                institution = sheet.cell_value(j+1,7)
                # institutions[i].add(institution)
            else:
                if (role == 'Co-I'):
                    institution = sheet.cell_value(j+1,6)
                    # institutions[i].add(institution)            
                else:
                    institution = sheet.cell_value(j+1,6)
            
            institution_short = abbreviate(institution)
            
            # XXX Add a line to print the proposal number and title here.

            line = (f'{role} {name_first} {name_last} / {institution_short}')
            if DO_LIST_FOR_GOOGLE:
                print(line)
            if DO_LIST_FOR_NICY:
                if (role == 'PI'):
                    print(f'{num_proposal} : {name_last} / {institution_short}')

        if DO_LIST_FOR_GOOGLE:
            print()
    print()
        
    # Now print a list of the most conflicted institutions on this panel    

    