# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import xlrd as xlrd

import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import os.path
import pickle

import random

import hbt_short as hbt

import re

def has_revote(record, column_min, column_max):
    """    
    Takes a record, and sees if there is a revote in it or not.
    """
    s = record[column_min : column_max]
    convert_first_to_generator = (str(item) for item in s)

    s2 = ' '.join(convert_first_to_generator)
    
    # Look for a paren, such as in "4.25 (4.57)", which indicates a revote
    # In the 2019 data, this is not present -- the original vote is not output.
    
    result = ('(' in s2) or (')' in s2)
    
    return result

def extract_vote_original(record, column):
    """
    Extract the original vote from a string like "4.00" or "4.25 (4.57)".
    """
    out = str(record[column]).split(' ')[0] 
    
    if (out == 'n/a'):
        out = 'nan'
        
    if len(out) == 0:  # Return 'nan' not '', since np.array.astype(float) won't choke on nan
        out = 'nan'
        
    return(out)

def extract_vote_revote(record, column):
    """
    Extract the revote from a string like "4.00" or "4.25 (4.57)".
    Assumes that no errors exist (ie, there actually is a revote)    
    """

    return( str(record[column]).split('(')[1].split(')')[0] )

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
    
file_xl = '/Users/hthroop/Downloads/SSW Trends.xls'

name_program = 'SSW'  # Search for this in the proposal number

workbook = xlrd.open_workbook(file_xl)
sheet_names = workbook.sheet_names()
print('Sheet Names', sheet_names)

num_years = len(sheet_names)

NamePI = []
NumberProposal = []
TitleProposal = []
NameInstitution = []
NameSubpanel = []
Notes = []
Week = []
NameSubpanel = []
Year = []

ScoreMeritMean = []
ScoreMeritMedian = []

ScoreCostMean = []
ScoreCostMedian = []

ScoreRelevanceMean = []
ScoreRelevanceMedian = []

ScorePMEMean = []
ScorePMEMedian = []

column_NameSubpanel    = 0
column_Week      = 1
column_NumberProposal  = 2
column_NamePI          = 3
column_TitleProposal   = 4
column_NameInstitution = 5
column_ScoreMeritMean  = 6
column_ScoreCostMean   = 8
column_ScoreRelevanceMean = 7
column_ScorePMEMean    = 9
column_ScoreMeritMedian  = 10
column_ScoreCostMedian   = 12
column_ScoreRelevanceMedian = 11
column_ScorePMEMedian  = 13
column_Notes           = 14   # In SSW19, we have 'Descope' and 'no PME' here as tags.

# Read in all the data

for i in range(num_years):
    
    sheet = workbook.sheet_by_index(i)
    num_rows = sheet.nrows
    
    for j in range(num_rows):
        record = sheet.row_values(j)
        if name_program in record[column_NumberProposal]:  # # If this is a a legit line
#            is_revote = '(' in record[column_ScoreMeritMean]
                
            NamePI.append(            record[column_NamePI])
            NameSubpanel.append(      record[column_NameSubpanel])
            TitleProposal.append(     record[column_TitleProposal])
            NumberProposal.append(    record[column_NumberProposal])
            NameInstitution.append(   record[column_NameInstitution])
            Notes.append(             record[column_Notes])
            Week.append(              record[column_Week])
            
            ScoreMeritMean.append(extract_vote_original(     record, column_ScoreMeritMean))
            ScoreCostMean.append(extract_vote_original(      record, column_ScoreCostMean))
            ScoreRelevanceMean.append(extract_vote_original( record, column_ScoreRelevanceMean))
            ScorePMEMean.append(extract_vote_original(       record, column_ScorePMEMean))

            ScoreMeritMedian.append(extract_vote_original(     record, column_ScoreMeritMedian))
            ScoreCostMedian.append(extract_vote_original(      record, column_ScoreCostMedian))
            ScoreRelevanceMedian.append(extract_vote_original( record, column_ScoreRelevanceMedian))
            ScorePMEMedian.append(extract_vote_original(       record, column_ScorePMEMedian))

### Convert all the data to NP arrays. This just lets us access them easier.
            
ScoreCostMean = np.array(ScoreCostMean).astype(float)
ScoreRelevanceMean = np.array(ScoreRelevanceMean).astype(float)
ScoreMeritMean = np.array(ScoreMeritMean).astype(float)

ScoreCostMedian = np.array(ScoreCostMedian).astype(float)
ScoreRelevanceMedian = np.array(ScoreRelevanceMedian).astype(float)
ScoreMeritMedian = np.array(ScoreMeritMedian).astype(float)

NamePI = np.array(NamePI)
NameLastPI = np.array(NamePI)
NameInstitution = np.array(NameInstitution)
NameSubpanel = np.array(NameSubpanel)  # Volcanism
NameSubpanelLong = np.array(NameSubpanel)  # 'SSW19 W3 Volcanism'
TitleProposal = np.array(TitleProposal)
NumberProposal = np.array(NumberProposal)
Notes = np.array(Notes)
Week = np.array(Week).astype(int)
NameSubpanel = np.array(NameSubpanel)

num_proposals = len(NamePI)

Year = np.zeros(num_proposals).astype(int)


### Extract just the PI last names, since SSW19 doesn't have first names

for i in range(len(NameLastPI)):
    s = NameLastPI[i]
    if ',' in s:
        s = s.split(',')[0]
        NameLastPI[i] = s

### Extract the year. This is the ROSES year, not the CY or FY
        
for i in range(num_proposals):
    Year[i] = NumberProposal[i][0:2]

# Make a long and unique string for each sub-panel, and tag it. 'SSW19 W4 Volcanism', for instance.

for i in range(num_proposals):
    NameSubpanelLong[i] = f'SSW{Year[i]} W{Week[i]} {NameSubpanel[i]}'

### Now time to make some plots!

bins_hist = np.arange(0, 5.15, 0.15)
plt.hist(ScoreMeritMedian,bins=bins_hist, label = 'Median')
plt.hist(ScoreMeritMean,bins=bins_hist, label = 'Mean', alpha=0.7)
plt.xlim([0.9, 5.1])
plt.xlabel('Merit Score')
plt.ylabel('# of Proposals')
plt.title(f'SSW14-18, N = {len(ScoreCostMean)} proposals')
plt.legend()
plt.show()

plt.plot(ScoreMeritMean, linestyle = 'none', marker='.', ms=1)
plt.show()

### And finally, now search for duplicate titles

num_proposals = len(TitleProposal)

SimilarityTitle = np.zeros([num_proposals, num_proposals])
SimilarityPI    = np.zeros([num_proposals, num_proposals])

# Loop through and calculate similarity for every pair of proposals. 
# Save these similarity metrics, but do not save the actual matches themselves, so we can change the criteria as needed.
# This is a slow process, so save the results to disk.,m                                                     
# (It is faster if I install a native package python-Levenshtein, but that requires gcc.
#   I can't install gcc, because *that* requires having Catalina, which *I don't on NASA Mac.)

file_pkl = f'{name_program}_{num_proposals}.pkl'

if os.path.exists(file_pkl):
     (SimilarityTitle, SimilarityPI) = pickle.load(open(file_pkl, "rb"))
     print(f'Loaded: {file_pkl}')
else:     
     
    for i in range(num_proposals):
        for j in range(num_proposals):
            SimilarityTitle[i,j] = fuzz.ratio(TitleProposal[i], TitleProposal[j])
            SimilarityPI[i,j]    = fuzz.ratio(NameLastPI[i],    NameLastPI[j])
            
        print(f'Done with {i}/{num_proposals}')
    print(f'Saving: {file_pkl}')
    pickle.dump( (SimilarityTitle, SimilarityPI), open( file_pkl, "wb" ) )

CriteriaSimilarityTitle = 75  # This is kind of arbitrary. 75 seems good.
CriteriaSimilarityPI    = 75  # SSW19 has surnames only, so relax matching here. 

# Now go through and search for matches.
# We end up with a list called 'matches', which indicates each proposal which matches the indicated one.
# Reversals are excluded (ie, if i == j, then j == i is excluded).
# Since some proposal ID's are listed twice, then explicitly count those as a non-match (e.g., main proposal, and PME).
# Also, exclude any that have PME, or descopes, or anything in the NOTES column.

matches = []

FILTER_STRING = 'SSW19'   # Matches anything on the string output

print('Searching for matches...')

## After matching a proposal, we need to take it out of the running. Don't double-count AB, AC, and CB.

AlreadyUsed = np.zeros(num_proposals).astype(bool)
AlreadyUsed[:] = False

for i in range(num_proposals):

    matches_i = [] 
    for j in range(num_proposals):
        if j > i:
            if ((SimilarityTitle[i,j] > CriteriaSimilarityTitle) and (SimilarityPI[i,j] > CriteriaSimilarityPI)
                and (NumberProposal[i] != NumberProposal[j]) 
                and (Notes[i] == '') 
                and (Notes[j] == '') 
#                and ( (FILTER_STRING in NumberProposal[j]) + (FILTER_STRING in NumberProposal[i]) )
                and not(AlreadyUsed[j])
                ):
                matches_i.append(j)        # Add the j element to list (ie, the 2nd, 3rd, 4th etc proposal)
                AlreadyUsed[j] = True      # Take it out of the running
                if len(matches_i) == 1:
                    matches_i.insert(0,i)  # Add the i element to list (ie, the 1st proposal)
                    print()
                    print("Matched!")
                print(f'{j:5} Ti: {SimilarityTitle[i,j]} {NumberProposal[j]:16} {NamePI[j]} {TitleProposal[j]}')
                print(f'{i:5} PI: {SimilarityPI[i,j]} {NumberProposal[i]:16} {NamePI[i]} {TitleProposal[i]}')
                print(f'Merit Mean: {ScoreMeritMean[matches_i]}')
                print()

# Now need to sort matches_i (in chronological order: 2014, then 2015, etc)
    order = np.argsort(matches_i)[::-1]   # Get the proper order            
    matches.append(list(np.array(matches_i)[order]))  # Add these new items to the list
        
# Now make a pretty list of all of the matched proposals

# And, make a list that we can plot
# If a proposal was submitted 2x, list as one point (one as Y1, and one as Y2).
# If a proposal was submitted 4x, list as three points       
# I am calling Y1 and Y2 here the first and second years of a pair. We will plot on X and Y axis, respectively.    
        
ScoreMeritMean_Y1   = []        
ScoreMeritMedian_Y1 = []        
ScoreMeritMean_Y2   = []        
ScoreMeritMedian_Y2 = []        


num_singles = 0                 # Number of proposals that were only submitted once
num_multiple_titles = 0         # Number of multiple-submits, counting each multiple only once
num_multiple_proposals = 0      # Number of multiple-submits, counting each individual submission

for prop in matches:
  if len(prop) > 0:
      num_multiple_titles += 1
      num_multiple_proposals += len(prop)
num_singles = len(matches) - num_multiple_proposals

print(f'{num_proposals} total proposals.')
print(f'  Single proposals: {num_singles}')
print(f'  Multiples, counting each multiple once: {num_multiple_titles}')
print(f'  Multiples, counting each individual submission: {num_multiple_proposals}')
print()

if FILTER_STRING:
    print(f'Listing proposals matching {FILTER_STRING}:')
    print()
      
for i in range(num_proposals):
    m = matches[i]
    if len(m) > 0:
#        print(f'{i}')
        out = ''
        for j in reversed(range(len(m))):
            delta = ScoreMeritMean[m[j]] - ScoreMeritMean[m[-1]]
            delta_str = f'{delta:+5.2f}'
            if '0.00' in delta_str:
                delta_str = '     '
            out = out + f'{NumberProposal[m[j]]:15} / W{Week[m[j]]:1} {NameSubpanel[m[j]][:20]:20}' + \
                  f'/ {NamePI[m[j]][:25]:25} ' + \
                  f' / {ScoreMeritMean[m[j]]:5.2f} {delta_str} / {TitleProposal[m[j]][:90]}\n'
        if FILTER_STRING in out:
            print(out)
#        print()
        
        scores_y1 = ScoreMeritMean[m[0:-1]]
        scores_y2 = ScoreMeritMean[m[1:]]
        
        ScoreMeritMean_Y1.extend(ScoreMeritMean[m[0:-1]])
        ScoreMeritMean_Y2.extend(ScoreMeritMean[m[1:  ]])
        
        ScoreMeritMedian_Y1.extend(ScoreMeritMedian[m[0:-1]])
        ScoreMeritMedian_Y2.extend(ScoreMeritMedian[m[1:  ]])

# And finally, make a plot!

ScoreMeritMean_Y1 = np.array(ScoreMeritMean_Y1)
ScoreMeritMean_Y2 = np.array(ScoreMeritMean_Y2)

plt.plot(ScoreMeritMean_Y1, ScoreMeritMean_Y2, ls='none', marker = '.', ms = 20, alpha=0.2)
plt.ylim([0,5])
plt.xlim([0,5])
plt.xlabel('Year 1')
plt.ylabel('Year 2')
plt.title('SSW Merit Mean, 2014-2019')
plt.plot([1,5],[1,5])
plt.gca().set_aspect('equal')
plt.show()

# See what the change would be if we just assigned Merit scores at random!
# This will change every time we run the code.

num_pairs = len(ScoreMeritMean_Y1)

ScoreMeritMean_Y1_s = np.array(random.choices(ScoreMeritMean, k=num_pairs))[0:-1]
ScoreMeritMean_Y2_s = np.array(random.choices(ScoreMeritMean, k=num_pairs))[0:-1]
delta_s = ScoreMeritMean_Y1_s - ScoreMeritMean_Y2_s

plt.hist(delta_s, bins=20, color = 'pink', alpha = 0.6)
plt.xlabel('Change in Synthetic Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW2014-2019. Delta mean = {np.nanmean(delta_s):4.2}; ' + 
                                        f'Stdev = {np.nanstd(delta_s):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.show()  

# Plot a histogram of the change

delta = ScoreMeritMean_Y2 - ScoreMeritMean_Y1
plt.hist(delta, bins=20, label = 'Actual')
plt.hist(delta_s, bins=20, alpha=0.6, color='pink', label = 'Random')
plt.xlabel('Change in Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW2014-2019. Delta mean = {np.mean(delta):4.2}; ' + 
                                        f'Stdev = {np.nanstd(delta):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.legend()
plt.show()    

# See what the most common institutions are

DO_LIST_INSTITUTIONS = False

if DO_LIST_INSTITUTIONS:
    institution = np.unique(NameInstitution) # Get a raw list of institutions. N = 179
    num_institutions = len(institution)
    
    indices_institution = []
    count_institution = []
    
    for i,inst in enumerate(list(institution)):
        indices_institution.append(np.where(inst == NameInstitution)[0])
        count_institution.append(len(indices_institution[-1]))
    count_institution = np.array(count_institution)
    indices_institution = np.array(indices_institution)
    
    # Make an ordered list of institutions, most popular at top
    
    order = np.argsort(count_institution)[::-1]  
    
    # Loop over all inst's, from top to bottom
    
    
    for i in range(num_institutions):
        inst_i = institution[order[i]]
        mean = np.nanmean(ScoreMeritMean[indices_institution[order[i]]])
        stdev = np.nanstd(ScoreMeritMean[indices_institution[order[i]]])
        
        print(f'{i+1:3}. {inst_i}')
        print(f' N = {count_institution[order[i]]}. Mean = {mean:4.3} +- {stdev:4.3}')
        
        print()
    
# See who the most busy PI's are

DO_LIST_PIS = True

if DO_LIST_PIS:
    PI = np.unique(NamePI) # Get a raw list of institutions. N = 179
    num_pis = len(PI)
    
    indices_pi = []
    count_pi = []
    
    for i,pi in enumerate(list(PI)):
        indices_pi.append(np.where(pi == NamePI)[0])
        count_pi.append(len(indices_pi[-1]))
        
    count_pi = np.array(count_pi)
    indices_pi = np.array(indices_pi)

# Make an ordered list of PI's, most popular at top

    order_pi = np.argsort(count_pi)[::-1]  
    
    # Loop over all inst's, from top to bottom
    
    for i in range(10):
        pi_i = PI[order_pi[i]]
        mean = np.nanmean(ScoreMeritMean[indices_pi[order_pi[i]]])
        stdev = np.nanstd(ScoreMeritMean[indices_pi[order_pi[i]]])
        
        print(f'{i+1:3}. {pi_i}')
        print(f' N = {count_pi[order_pi[i]]}. Mean = {mean:4.3} +- {stdev:4.3}')
        print()

DO_LIST_PANELS = True

# Get a list of all of the panels
# Make a plot of the mean score on each panel

Panel_RankOrder = {}   # Dictionary. So RankOrder['SSW19 Volcanism'] will return [223, 112, 2070, 1], etc.
Panel_Mean = {}

NamesSubpanelLong = np.unique(NameSubpanelLong)
ScoreSubpanelMean = []

for panel in NamesSubpanelLong:
    w = np.where(panel == NameSubpanelLong)[0]
    m = np.nanmean(ScoreMeritMean[w])
    ScoreSubpanelMean.append(m)
    
    print(f'{panel:40} {len(w):3} {m:5.2f}')
ScoreSubpanelMean = np.array(ScoreSubpanelMean)

hbt.fontsize(24)
plt.hist(ScoreSubpanelMean,bins=20)
plt.xlabel(f'Subpanel Average of Mean Merit')
plt.ylabel('Number of Subpanels')
plt.title(f'Subpanel Mean; N = {len(NamesSubpanelLong)} SSW Subpanels')
plt.ylim([0,24])
plt.show()
    
# Calculate the rank and percentile rank of each proposal, on its respective panel
    
    
        