# -*- coding: utf-8 -*-
"""
This is my code to analyze repeat proposal scores. Written Spring-2020, NASA HQ, HBT.

Code reads in Excel files with scoresheets in a standard format.

Code reads in a list of selections, in a CSV file.

Code does *not* read in the entire dump from NSPIRES (e.g., budgets). It should be adapted to do so.

This was the first real code I wrote in Python after a hiatus. It is not that pythonic. Mostly procedural, and
it does not use functions or dictionaries very well.

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
import csv
from scipy import stats   # For lin regress

from scipy.stats.stats import pearsonr

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

##########
### Start of Main Code
##########
    
# Define the file inputs
    
path_base = '/Users/hthroop/Documents/HQ/SSW/'    
file_xl = 'SSW Trends.xls'  # Scoresheets
file_selections = 'Selections_SSW.csv' # Text file with selection values (Yes/No)

# Define the plot settings

hbt.fontsize(12)
figsize_default = (7,5)
hbt.figsize(figsize_default)

name_program = 'SSW'  # Search for this in the proposal number

workbook = xlrd.open_workbook(os.path.join(path_base, file_xl))

sheet_names = workbook.sheet_names()
print('Sheet Names', sheet_names)


sheet_names = sheet_names[0:2]

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
column_NameInstitution = 5   # Missing in 2019 data
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

IsSelected = np.zeros(num_proposals).astype(bool)

### Extract just the PI last names, since SSW19 doesn't have first names

for i in range(len(NameLastPI)):
    s = NameLastPI[i]
    if ',' in s:
        s = s.split(',')[0]
        NameLastPI[i] = s

### Extract the year. This is the ROSES year, not the CY or FY
        
for i in range(num_proposals):
    Year[i] = NumberProposal[i][0:2]

### Read in the list of selections (i.e., select vs. decline)
### The only hitch here is we need to change format from SSW19-100 to SSW19-0100 (ugh...)    
    
file_selections = os.path.join(path_base, file_selections)
with open(file_selections, newline='') as csvfile:
    lines = csv.reader(csvfile, delimiter=',', quotechar='"') 
    for line in lines:
        if 'Select' in line[1]:
            proposal = line[0]
            parts = proposal.split('-')
            proposal2 = f'{parts[0]}-{parts[1]}-{int(parts[2]):04d}'
            proposal = proposal2
            i = np.where(NumberProposal == proposal)[0]
            IsSelected[i] = True
            # print(f'{NumberProposal[i]} Selected***, proposal={proposal}, i={i}')
        # else:
            # print(f'{NumberProposal[i]} Declined, i={i}')

IsDeclined = np.logical_not(IsSelected)
           
# Make a long and unique string for each sub-panel, and tag it. 'SSW19 W4 Volcanism', for instance.

for i in range(num_proposals):
    NameSubpanelLong[i] = f'SSW{Year[i]} W{Week[i]} {NameSubpanel[i]}'

### And finally, now search for duplicate titles

num_proposals = len(TitleProposal)

SimilarityTitle = np.zeros([num_proposals, num_proposals])
SimilarityPI    = np.zeros([num_proposals, num_proposals])

# Loop through and calculate similarity for every pair of proposals. 
# Save these similarity metrics, but do not save the actual matches themselves, so can change the criteria as needed.
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

FILTER_STRING = 'Umur'   # Matches anything on the string output

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
                print(f'{j:5} // {SimilarityTitle[i,j]} // {NumberProposal[j]:16} // {NamePI[j]} // {TitleProposal[j]}')
                print(f'{i:5} // {SimilarityPI[i,j]}    // {NumberProposal[i]:16} // {NamePI[i]} // {TitleProposal[i]}')
                print(f'Merit Mean: {ScoreMeritMean[matches_i]}')
                print()

# Now need to sort matches_i (in chronological order: 2014, then 2015, etc)
    order = np.argsort(matches_i)[::-1]   # Get the proper order            
    matches.append(list(np.array(matches_i)[order]))  # Add these new items to the list

# Extract the sequence number of each proposal (-0001, etc)
# This lets us see if there is a trend front start to end.    
# Also, get the %ile (i.e., 1.0 = final proposal submitted)
    
NumberProposalSequential = []
NumberProposalSequentialPercentile = []

for id in NumberProposal:
    (y,_,s) = id.split('-')
    s = int(s)
    y = int(y)
    NumberProposalSequential.append(s)
    p = s / np.sum(Year == y)
NumberProposalSequential = np.array(NumberProposalSequential).astype(int)   

for id in NumberProposal:
    (y,_,s) = id.split('-')
    s = int(s)
    y = int(y)
    p = s / np.max(NumberProposalSequential[Year == y])
    NumberProposalSequentialPercentile.append(p)
NumberProposalSequentialPercentile = np.array(NumberProposalSequentialPercentile)  


################################
### Now time to make some plots!
################################

color_decline = 'red'
color_select = 'green'

hbt.fontsize(12)

bins_hist = np.arange(0, 5.15, 0.15)
plt.hist(ScoreMeritMedian,bins=bins_hist, label = 'Median')
plt.hist(ScoreMeritMean,bins=bins_hist, label = 'Mean', alpha=0.7)
plt.xlim([0.9, 5.1])
plt.xlabel('Merit Score')
plt.ylabel('# of Proposals')
plt.title(f'SSW14-18, N = {len(ScoreCostMean)} proposals')
plt.legend()
plt.show()

# Make a histogram of Select and Decline

plt.hist(ScoreMeritMean[IsSelected],bins=bins_hist, label = 'Select', color=color_select, alpha=0.5)
plt.hist(ScoreMeritMean[np.logical_not(IsSelected)],bins=bins_hist, label = 'Decline', color=color_decline, alpha=0.2)
plt.xlim([0.9, 5.1])
plt.xlabel('Merit Score')
plt.ylabel('# of Proposals')
plt.title(f'SSW14-18, N = {len(ScoreCostMean)} proposals')
plt.legend()
plt.show()

n = np.arange(num_proposals)

plt.plot(n[IsDeclined], ScoreMeritMean[IsDeclined], linestyle = 'none', marker='.', ms=3, color=color_decline,
         alpha=0.7)
plt.plot(n[IsSelected], ScoreMeritMean[IsSelected], linestyle = 'none', marker='.', ms=5, color=color_select, 
         alpha=0.7)
plt.xlabel('Sequential Proposal Number [since 2014]')
plt.ylabel('Merit Score, Mean')
plt.show()

plt.plot(n[IsDeclined], ScoreMeritMedian[IsDeclined], linestyle = 'none', marker='.', ms=3, color=color_decline, 
         alpha=0.7)
plt.plot(n[IsSelected], ScoreMeritMedian[IsSelected], linestyle = 'none', marker='.', ms=5, color=color_select, 
         alpha=0.7)
plt.xlabel('Sequential Proposal Number [since 2014]')
plt.ylabel('Merit Score, Median')
plt.show()

plt.plot(ScoreMeritMean[IsDeclined], ScoreMeritMedian[IsDeclined], linestyle = 'none', marker='.', ms=10, 
         color=color_decline, alpha=0.3)
plt.plot(ScoreMeritMean[IsSelected], ScoreMeritMedian[IsSelected], linestyle = 'none', marker='.', ms=10, 
         color=color_select, alpha=0.3)
plt.title('Mean vs. Median')
plt.xlabel('Merit Score, Mean')
plt.ylabel('Merit Score, Median')
plt.plot([0,4],[4,4], alpha=0.1, color='black')
plt.plot([4,4],[0,4], alpha=0.1, color='black')
plt.gca().set_aspect('equal')
plt.xlim([0.9,5.2])
plt.ylim([0.9,5.2])
plt.show()


#%%

# Make a plot of the mean score on each panel

DO_LIST_PANELS = True

Panel_RankOrder = {}   # Dictionary. So RankOrder['SSW19 Volcanism'] will return [223, 112, 2070, 1], etc.
Panel_Mean = {}

NamesSubpanelLong = np.unique(NameSubpanelLong)
ScoreSubpanelMean = []                          # Mean score on this subpanel 
ScoreSubpanelMin  = []                          # Mean score on this subpanel 
ScoreSubpanelMax  = []                          # Mean score on this subpanel 
CountSubpanel     = []                          # Number of proposals on this sub-panel

for panel in NamesSubpanelLong:
    w = np.where(panel == NameSubpanelLong)[0]

    m = np.nanmean(ScoreMeritMean[w])
    
    ScoreSubpanelMean.append(np.nanmean(ScoreMeritMean[w]))
    ScoreSubpanelMin.append(np.nanmin(ScoreMeritMean[w]))
    ScoreSubpanelMax.append(np.nanmax(ScoreMeritMean[w]))
                            
    CountSubpanel.append(len(w))
    
    print(f'{panel:40}/{len(w):3}/{m:5.2f}')
    
ScoreSubpanelMean = np.array(ScoreSubpanelMean)
ScoreSubpanelMin = np.array(ScoreSubpanelMin)
ScoreSubpanelMax = np.array(ScoreSubpanelMax)
CountSubpanel     = np.array(CountSubpanel)

plt.hist(ScoreSubpanelMean,bins=20)
plt.xlabel(f'Subpanel Average of Mean Merit')
plt.ylabel('Number of Subpanels')
plt.title(f'Subpanel Mean; N = {len(NamesSubpanelLong)} SSW Subpanels')
plt.ylim([0,24])
plt.show()

# Make a 'Normalized Merit Score' for each proposal. 
# To do this, divide the proposal's score by the panel mean, then mult x5.
#
# My guess is that this could be important. If we find a proper mapping from score
# to normalized score, we might be able to de-bias the score, and do well.

ScoreMeritMeanNorm = np.zeros(num_proposals)

for i in range(num_proposals):
    panel = NameSubpanelLong[i]
    mean_subpanel = ScoreSubpanelMean[np.where(NamesSubpanelLong == panel)[0]][0]
    max_subpanel = ScoreSubpanelMax[np.where(NamesSubpanelLong == panel)[0]][0]
    min_subpanel = ScoreSubpanelMin[np.where(NamesSubpanelLong == panel)[0]][0]
    
    # score_norm = ScoreMeritMean[i] / mean_subpanel * 5  # This is one possible metric. Scale by panel mean.

    # Scale every panel so it uses the full dynamic range 1..5
    
    score_norm = 1 + ((ScoreMeritMean[i] - min_subpanel) / (max_subpanel-min_subpanel) * 4) 
    
    ScoreMeritMeanNorm[i] = score_norm # This is some sort of normalized merit score

plt.plot(ScoreMeritMean, ScoreMeritMeanNorm, marker='.', ls='none')
plt.xlabel('Merit Mean [raw]')
plt.ylabel('Merit Mean [normalized]')
plt.show()

plt.plot(ScoreMeritMean[IsDeclined], ScoreMeritMeanNorm[IsDeclined], marker='.', ls='none', 
         label='Declined', color=color_decline, alpha=0.2)
plt.plot(ScoreMeritMean[IsSelected], ScoreMeritMeanNorm[IsSelected], marker='.', ls='none', 
         label='Selected', color=color_select, alpha=0.2)
plt.xlabel('Merit Mean [raw]')
plt.ylabel('Merit Mean [normalized]')
plt.legend()
plt.show()
    
# Calculate the rank and percentile rank of each proposal, on its respective panel

RankOnSubpanel = np.zeros(num_proposals).astype(int)
PercentileOnSubpanel = np.zeros(num_proposals).astype(int)

for panel in NamesSubpanelLong:
    w = np.where(panel == NameSubpanelLong)[0]  # Get all the proposals on this subpanel
    o = np.argsort(ScoreMeritMean[w])
    r = o.argsort()
    r = np.amax(r) - r   # Reverse it, so rank is from top down]
    PercentileOnSubpanel[w] = 100 - r / np.max(r) * 100
    RankOnSubpanel[w] = r+1

#%%
    
# List the proposals on each subpanel, by score / rank / %ile

DO_HIST_EVERY_SUBPANEL = True

FILTER = 'SSW19'

hbt.figsize(25,30)
ii = 1
hbt.fontsize(20)    
for panel in NamesSubpanelLong:
    if FILTER in panel:
        if (FILTER == 'SSW19'):
            plt.subplot(6,4,ii)    # Make a matrix of panels. Fine-tune this as needed.
        w = np.where(panel == NameSubpanelLong)[0]  # Get all the proposals on this subpanel
        o = np.argsort(ScoreMeritMean[w])[::-1]
        print(f'{panel}')
        for  i in w[o]:
            star = np.array([" ","*"])[1*IsSelected[i]]
            print(f'{i:4} {RankOnSubpanel[i]:3} {PercentileOnSubpanel[i]:4} {star}' +\
                  f'{ScoreMeritMean[i]:6}' + \
                  f'  {NamePI[i][0:12]:12}  {TitleProposal[i]} ')
        print()    
    
        if DO_HIST_EVERY_SUBPANEL:
            # plt.hist(ScoreMeritMean[w])
            # plt.title(panel)
            # plt.xlim([1,5])
            # plt.ylim([0,7])
            # plt.show()
            
            indices_selected = w * IsSelected[w]
            indices_declined = w * IsDeclined[w]
            indices_selected = indices_selected[indices_selected > 0]
            indices_declined = indices_declined[indices_declined > 0]
            
            # Define the histograph bins. Need to be careful to start bins on half-border, and go well beyond top.
            
            bins = np.arange(-0.25, 5.5, 0.5)  
            
            # plt.hist(ScoreMeritMean[indices_selected], color=color_select, alpha=0.5, bins=bins)
            plt.hist(ScoreMeritMean[indices_selected], color=color_select, alpha=0.5, bins=bins)
            plt.hist(ScoreMeritMean[indices_declined], color=color_decline, alpha=0.1, bins=bins)
            plt.title(panel.replace('SSW19', ''))
            plt.xlim([0.5,5.5])
            plt.ylim([0,9])
            # plt.show()
            ii += 1

plt.tight_layout()
plt.show()   
hbt.fontsize(12)  # Undo fontsize change from above (one-off)

hbt.figsize(figsize_default)

#%%
# Make a plot of rank vs. score

plt.plot(ScoreMeritMean, RankOnSubpanel, marker='.', ls='none')
plt.xlabel('Score Merit Mean')
plt.ylabel('Rank on Panel') 
plt.show()  

#%%
# Plot %ile on panel

plt.plot(ScoreMeritMean, PercentileOnSubpanel, marker='.', ls='none')
plt.xlabel('Score Merit Mean')
plt.ylabel('Percentile on Panel') 
plt.show()  

plt.plot(ScoreMeritMean[IsSelected], PercentileOnSubpanel[IsSelected], 
         label = 'Selected', marker='.', ls='none',color=color_select, alpha=0.2)
plt.plot(ScoreMeritMean[IsDeclined], PercentileOnSubpanel[IsDeclined], 
         label = 'Declined', marker='.', ls='none',color=color_decline, alpha=0.2)
plt.legend()
plt.xlabel('Score Merit Mean')
plt.ylabel('Percentile on Panel') 
plt.show() 

#%%        
# Now make a pretty list of all of the matched proposals

# And, make a list that we can plot
# If a proposal was submitted 2x, list as one point (one as Y1, and one as Y2).
# If a proposal was submitted 4x, list as three points       
# I am calling Y1 and Y2 here the first and second years of a pair. We will plot on X and Y axis, respectively.    
        
ScoreMeritMean_Y1   = []        
ScoreMeritMedian_Y1 = []        
ScoreMeritMean_Y2   = []        
ScoreMeritMedian_Y2 = []
PercentileOnSubpanel_Y1 = []
PercentileOnSubpanel_Y2 = []

ScoreMeritMeanNorm_Y1 = []
ScoreMeritMeanNorm_Y2 = []

IsSelected_Y2 = []
IsResubmit = np.zeros(num_proposals).astype(bool)

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

trends = []  # Set up a list of score trends. This is just an array of arrays.

# FILTER_STRING = 'Dela'      
for i in range(num_proposals):
    m = matches[i]
    if len(m) > 0:
#        print(f'{i}')
        out = ''
        trend = []
        for j in reversed(range(len(m))):   # Now loop over all proposals in the match
            if j == len(m)-1:
                delta = 0  # List the delta from the previous score, except for the first one
            else:    
                delta = ScoreMeritMean[m[j]] - ScoreMeritMean[m[j-1]]
            delta_str = f'{delta:+5.2f}'
            if '0.00' in delta_str:  # Skip the delta for the first proposal in a match
                delta_str = '     '
            else:
                IsResubmit[m[j]] = True # If this is a resubmit, flag as so
                
            if IsSelected[m[j]]:
                select_str = 'Select'
            else:
                select_str = ''
            out = out + f'{NumberProposal[m[j]]:15} / W{Week[m[j]]:1} {NameSubpanel[m[j]][:20]:20}' + \
                  f'/ {NamePI[m[j]][:25]:25} ' + \
                  f' / {ScoreMeritMean[m[j]]:5.2f} {delta_str} / {select_str:8} ' + \
                  f' / {RankOnSubpanel[m[j]]:3} = {PercentileOnSubpanel[m[j]]:3}% / ' + \
                  f'{TitleProposal[m[j]][:90]}\n'
            trend.append(ScoreMeritMean[m[j]])
        trends.append(trend)
        
        if FILTER_STRING in out:
            print(out)
        
        ScoreMeritMean_Y2.extend(ScoreMeritMean[m[0:-1]])
        ScoreMeritMean_Y1.extend(ScoreMeritMean[m[1:  ]])
        
        ScoreMeritMeanNorm_Y2.extend(ScoreMeritMeanNorm[m[0:-1]])
        ScoreMeritMeanNorm_Y1.extend(ScoreMeritMeanNorm[m[1:  ]])
        
        ScoreMeritMedian_Y2.extend(ScoreMeritMedian[m[0:-1]])
        ScoreMeritMedian_Y1.extend(ScoreMeritMedian[m[1:  ]])

        PercentileOnSubpanel_Y2.extend(PercentileOnSubpanel[m[0:-1]])
        PercentileOnSubpanel_Y1.extend(PercentileOnSubpanel[m[1:]])
        
        IsSelected_Y2.extend( IsSelected[m[0:-1]] )

IsSelected_Y2 = np.array(IsSelected_Y2)
IsDeclined_Y2 = np.logical_not(IsSelected_Y2)

ScoreMeritMedian_Y1 = np.array(ScoreMeritMedian_Y1)
ScoreMeritMedian_Y2 = np.array(ScoreMeritMedian_Y2)
ScoreMeritMean_Y1 = np.array(ScoreMeritMean_Y1)
ScoreMeritMean_Y2 = np.array(ScoreMeritMean_Y2)
ScoreMeritMeanNorm_Y1 = np.array(ScoreMeritMeanNorm_Y1)
ScoreMeritMeanNorm_Y2 = np.array(ScoreMeritMeanNorm_Y2)
PercentileOnSubpanel_Y1 = np.array(PercentileOnSubpanel_Y1)
PercentileOnSubpanel_Y2 = np.array(PercentileOnSubpanel_Y2)

   
#%%  
# And finally, make a plot of Y1 vs Y2 score. This is the money plot.

# color_select = 'blue'
# color_decline = 'blue'

ScoreMeritMean_Y1 = np.array(ScoreMeritMean_Y1)
ScoreMeritMean_Y2 = np.array(ScoreMeritMean_Y2)

plt.plot(ScoreMeritMean_Y1[IsDeclined_Y2], ScoreMeritMean_Y2[IsDeclined_Y2], 
         ls='none', marker = '.', ms = 20, 
         alpha=0.2, color=color_decline)

plt.plot(ScoreMeritMean_Y1[IsSelected_Y2], ScoreMeritMean_Y2[IsSelected_Y2], 
         ls='none', marker = '.', ms = 15, alpha=0.3, color=color_select)

r = pearsonr(ScoreMeritMean_Y1, ScoreMeritMean_Y2)[0]
plt.ylim([0.9,5.1])
plt.xlim([0.9,5.1])
plt.xlabel('Merit Score, Year 1')
plt.ylabel('Merit Score, Year 2')
plt.title(f'SSW Merit Mean, 2014-2019')
plt.plot([1,5],[1,5])
plt.gca().set_aspect('equal')
plt.show()


#%%
# Plot Normalized score: Y1 vs. Y2

plt.plot(ScoreMeritMeanNorm_Y1[IsDeclined_Y2], ScoreMeritMeanNorm_Y2[IsDeclined_Y2], 
         ls='none', marker = '.', ms = 20, alpha=0.2, color=color_decline)
plt.plot(ScoreMeritMeanNorm_Y1[IsSelected_Y2], ScoreMeritMeanNorm_Y2[IsSelected_Y2], 
         ls='none', marker = '.', ms = 20, alpha=0.2, color=color_select)

r = pearsonr(ScoreMeritMeanNorm_Y1, ScoreMeritMeanNorm_Y2)[0]
plt.ylim([0.9, 5.1])
plt.xlim([0.9, 5.1])
plt.xlabel('Merit Score, Year 1')
plt.ylabel('Merit Score, Year 2')
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3, 4, 5])
plt.title(f'SSW Merit Mean Normalized (1-5), SSW14-19')
plt.plot([1,100],[1,100])
plt.gca().set_aspect('equal')
plt.show()

#%%
# Plot Y1 %ile vs. Y2 %ile
# Conclusion: I thought this would be useful, but it's totally not.
# I thought it would bounce around a bit, but a proposal near the top would stay near the top. Nope!

plt.plot(PercentileOnSubpanel_Y1[IsSelected_Y2], PercentileOnSubpanel_Y2[IsSelected_Y2], ls='none', 
         marker = '.', ms = 20, alpha=0.2, color=color_select, label = 'Select')
plt.plot(PercentileOnSubpanel_Y1[IsDeclined_Y2], PercentileOnSubpanel_Y2[IsDeclined_Y2], ls='none', 
         marker = '.', ms = 20, alpha=0.2, color=color_decline, label = 'Decline')

r = pearsonr(PercentileOnSubpanel_Y1, PercentileOnSubpanel_Y2)[0]
plt.ylim([-5,105])
plt.xlim([-5,105])
plt.xlabel('Merit %ile, Year 1')
plt.ylabel('Merit %ile, Year 2')
plt.title(f'SSW Merit Mean, %ile on Subpanel, SSW14-19')
plt.plot([1,100],[1,100])
plt.gca().set_aspect('equal')
plt.show()

#%%
# Plot the trends -- one 'squiggle' line for each individual proposal and its resubmits

num_trends = len(trends)
num_columns = 6
width_column = 5
lines_per_column = int(num_trends / num_columns)
alpha = 0.85

for i,t in enumerate(trends):
    
    delta = t[1] - t[0]
    
    if (delta > 0):
        color = 'PaleGreen'
    if (delta > 1):
        color = 'Lime'
    if (delta > 2):
        color = 'Green'
    if (delta < 0):
        color = 'Orange'
    if (delta < -1):
        color = 'Red'
    if (delta < -2):
        color = 'Crimson'
    if np.abs(delta) < 0.5:
        color = 'Grey'
        
    colnum = int(num_columns * i/num_trends)
    x0 = np.arange(len(t)) + colnum*width_column
    y0 = np.mod(i,lines_per_column)
    plt.plot(x0, np.array(t)+y0, color=color, alpha=alpha)
    plt.xlim([-5,num_columns*(width_column)+2])
    plt.ylim([0,lines_per_column+5])
    plt.title('Year-To-Year Score Trend for Every SSW Resubmitted Proposal')

# Manually build a legend, showing how the slope of each line indicates its score change.
    
x0 = -4
x1 = -3

for i,delta in enumerate([-3, -2, -1, 0, 1, 2, 3]):
    y0 = lines_per_column/2 + i
    y1 = y0 + delta

    color=['Crimson', 'Red', 'Orange', 'Grey', 'PaleGreen', 'Lime', 'Green'][i]

    plt.plot([x0,x1], [y0, y1], color=color, alpha=alpha)
    plt.text(x1+0.25, y1, f'{delta:+}', fontsize=7, color='black')
    
    print(f'i={i}, delta={delta}, color={color}')

plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.show()


#%%
    
# See what the change would be if we just assigned Merit scores at random!
# This will change every time we run the code.

# _s = extension for SYNTHETIC (i.e., random) Y2 values
# _l = extension for LONG Y2 values (ie, more cases than actual)

bins = np.arange(-4, 4, 0.25)

FACTOR = 1000 # When making the random distribution, increase the # of drawings by this much, to get a smooth curve.

num_pairs = len(ScoreMeritMean_Y1)-2
pool_nonan = ScoreMeritMean_Y1.copy()

# Make the random 'synthetic' values (normal length)
ScoreMeritMean_Y1_s = np.array(random.choices(ScoreMeritMean_Y1, k=len(ScoreMeritMean_Y1)))
ScoreMeritMean_Y2_s = np.array(random.choices(ScoreMeritMean_Y2, k=len(ScoreMeritMean_Y2)))
delta_s = ScoreMeritMean_Y2_s - ScoreMeritMean_Y1_s

# Make the 'long' synthetic values, which give better statistics

ScoreMeritMean_Y1_sl = np.array(random.choices(ScoreMeritMean_Y1, k=FACTOR*len(ScoreMeritMean_Y1)))
ScoreMeritMean_Y2_sl = np.array(random.choices(ScoreMeritMean_Y2, k=FACTOR*len(ScoreMeritMean_Y2)))
delta_sl = ScoreMeritMean_Y2_sl - ScoreMeritMean_Y1_sl

#  Calculate the histogram, and grab the output. We don't keep the plot itself, 
# because we want to rescale it and replot next to the original data.

(hist_n, hist_bins, hist_patches) = plt.hist(delta_sl, bins=bins, color = 'pink', alpha = 0.6)
hist_n /= FACTOR  # Scale the results back down.

plt.bar(hist_bins[0:-1], hist_n)

plt.xlabel('Change in Synthetic Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW14-19. Delta mean = {np.nanmean(delta_s):4.2}; ' + 
                                        f'Stdev = {np.nanstd(delta_s):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.show()  

#%%

# Now that we have generated the random scores, make a side-by-side scatter plot of scores vs. randomized scores

plt.subplot(1, 2, 1)
plt.plot(ScoreMeritMean_Y1, ScoreMeritMean_Y2, ls='none', marker='.', 
         alpha=0.20, ms=10)
plt.ylim([0.5,5.1])
plt.xlim([0.5,5.1])
plt.xlabel('Y1 Merit')
plt.ylabel('Y2 Merit [Actual]')
plt.gca().set_aspect('equal')
plt.subplot(1,2,2)
plt.plot(ScoreMeritMean_Y1, ScoreMeritMean_Y2_s, ls='none', marker='.', 
         alpha=0.20, ms=10)
plt.ylim([0.9,5.1])
plt.xlim([0.9,5.1])
plt.xlabel('Y1 Merit')
plt.ylabel('Y2 Merit [Randomized]')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

#%%

# Plot a histogram of the change

delta = ScoreMeritMean_Y2 - ScoreMeritMean_Y1

dev    = np.nanstd(delta)
dev_sl = np.nanstd(delta_sl)
m      = np.nanmean(delta)
m_sl   = np.nanmean(delta_sl)

# plt.bar(hist_bins[0:-1], hist_n, alpha=0.5, color='pink', label='Randomized scores', width=0.25)
plt.hist(delta, label = f'Actual scores, N={num_pairs}', bins=bins, alpha=0.25, color='blue')

plt.xlabel('Change in Merit Score')
plt.ylabel('Number')
plt.title(f'SSW14-19. Score change for resubmitted proposal')  
plt.axvline(0, color='red', alpha=0.2)
plt.legend()
plt.show()    

print(f'Actual     scores: Expected change = {m:5.2f} +- {dev:5.2f}')
print(f'Randomized scores: Expected change = {m_sl:5.2f} +- {dev_sl:5.2f}')

#%%

# As per suggestion by MB, plot score, vs. expected change in score.

slope, intercept, r_value, p_value, std_err = stats.linregress(ScoreMeritMean_Y1, delta)
x = np.arange(1, 6, 1)
y = slope*x + intercept

plt.plot(ScoreMeritMean_Y1[IsDeclined_Y2], delta[IsDeclined_Y2], ls='none', marker='.', 
         label = 'Declined', color=color_decline)
plt.plot(ScoreMeritMean_Y1[IsSelected_Y2], delta[IsSelected_Y2], ls='none', marker='.', 
         label = 'Selected', color=color_select)
plt.xlabel('Y1 Merit Score')
plt.ylabel('Change in Merit, Y2-Y1')
plt.plot(x, y, color='orange', label = 'Best Fit')
plt.plot([1,5],[0,0], color='black', alpha=0.3, label = 'No Change')
mm = np.nanmean(ScoreMeritMean_Y1)
# plt.plot([mm,mm], [-3,3], color='black', alpha=0.3)
plt.ylim([-3,3])
plt.xlim([0.9, 5.1])
plt.legend()
plt.show()

# Do the same, but for the synthetic data set

delta_s = ScoreMeritMean_Y2_s - ScoreMeritMean_Y1
slope, intercept, r_value, p_value, std_err = stats.linregress(ScoreMeritMean_Y1, delta_s)
x = np.arange(1, 6, 1)
y = slope*x + intercept

plt.plot(ScoreMeritMean_Y1, delta_s, ls='none', marker='.', label = 'SSW')
plt.xlabel('Y1 Merit Score [Actual]')
plt.ylabel('Change in Merit, Y2-Y1 [Randomly Drawn]')
plt.plot(x, y, color='orange', label = 'Best Fit')
plt.plot([1,5],[0,0], color='black', alpha=0.3, label = 'No Change')
mm = np.nanmean(ScoreMeritMean_Y1)
# plt.plot([mm,mm], [-3,3], color='black', alpha=0.3)
plt.ylim([-3,3])
plt.xlim([0.9, 5.1])
plt.legend()
plt.show()

#%% 
# Make a plot of Score vs. Submission Order

plt.plot(NumberProposalSequentialPercentile, ScoreMeritMean, ls='none', marker='.', alpha = 0.5)
plt.xlabel('Submission Order Within Program Year (0 = First    1 = Last)')
plt.ylabel('Merit Mean Score')
plt.show()

#%%
# Make a plot of change in score, vs. score.

# Plot a histogram of the change, color-coded for select v decline

plt.hist(delta[IsDeclined_Y2], color = color_decline, label='Decline', alpha = 0.6, bins=bins)
plt.hist(delta[IsSelected_Y2], color = color_select, label='Select', alpha = 0.6, bins=bins)
plt.legend()
plt.xlabel('Change in Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW2014-2019. Delta mean = {np.nanmean(delta_s):4.2}; ' + 
                                        f'Stdev = {np.nanstd(delta_s):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.show()  

#%%
# See what the most common institutions are

DO_LIST_INSTITUTIONS = True

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
        
        # List the number per year
        
        for y in np.unique(Year):
            n = np.sum(np.logical_and( NameInstitution == inst_i, Year == y))
            print(f' {y}: N = {n}')
        
        print()

#%%    
# See who the most busy PI's are
# This does not use robust PI-matching, so it's not really totally useful.
# e.g., Sven Simon shows up with three different names.

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
    
    for i in range(300):
        pi_i = PI[order_pi[i]]
        mean = np.nanmean(ScoreMeritMean[indices_pi[order_pi[i]]])
        stdev = np.nanstd(ScoreMeritMean[indices_pi[order_pi[i]]])
        n_select = np.sum(IsSelected[indices_pi[order_pi[i]]])
        
        print(f'{i+1:3}. {pi_i:25} N = {n_select} / {count_pi[order_pi[i]]} = ' + \
              f'{100*n_select / count_pi[order_pi[i]]:3.0f}%.   Mean = {mean:4.3} +- {stdev:4.3}' )

#%%
            
# Make a plot of quartile vs. quartile

#%%
            
# Make a plot of Y1 vs. Y2 for each year. So, for 2014 vs. 2015, 18-19, etc. This will not be as large, but
# will identify any changdes in the program (e.g., Mary Voytek → Delia).
# Any year-by-year statistical info we should put here, too.
            
DO_ANNUAL_TABLE = True
DO_INCLUDE_RESUBMITS = False  # Include these in the plot? Always included in table.

i = 1
print('Program  N    N_Resub  (%)     N_select  (%)    N_Select_of_Resub (%)  Mean  Mean_of_Resub')

for y in np.unique(Year):
    plt.subplot(2,3,i)
    bins = np.arange(1-0.25/2, 5.2, 0.25)
    indices = np.where(Year == y)[0]
    plt.hist(ScoreMeritMean[indices], bins=bins, label='Total')
    
    indices2 = np.where( np.logical_and(Year==y, IsResubmit == True))[0]

    num_sub      = np.sum(Year == y)
    num_sub_r    = np.sum(np.logical_and( (Year == y), (IsResubmit==True)))
    num_select   = np.sum(np.logical_and( (Year == y), (IsSelected == True)))
    num_select_r = np.sum(np.logical_and( (Year == y), np.logical_and((IsSelected == True), (IsResubmit==True))))
    
    plt.title(f'SSW{y}, N={len(indices)}')
    plt.xticks([1, 2, 3, 4, 5])

    if DO_INCLUDE_RESUBMITS:
        plt.hist(ScoreMeritMean[indices2], label='Resubmit', alpha=0.5, bins=bins)
        plt.ylabel('N')
        plt.xlabel('Score')
        if (y == min(Year)):
            plt.legend()
    plt.tight_layout()
    i += 1
    
    m = np.nanmean(ScoreMeritMean[indices])
    m_r = np.nanmean(ScoreMeritMean[indices2])
    frac = len(indices2) / len(indices)
    
    print(f'SSW{y}: N={len(indices):3}  N_r={len(indices2):3} ({frac*100:2.0f}%)' +
          f'  N_s = {num_select:3} ({100*num_select/num_sub:2.0f}%)  ' + 
          f'  N_s_r = {num_select_r:3} ({100*num_select_r/num_sub_r:2.0f}%)  Mean={m:5.3}  Mean_r={m_r:5.3}')

plt.show()    

#%%           
# Make a histogram of all submits vs. totals

bins = np.arange(1-0.25/2, 5.2, 0.25)

indices = np.where( IsResubmit == True)[0]

plt.hist(ScoreMeritMean, label=f'Total, $N$={num_proposals}', bins=bins)

if DO_INCLUDE_RESUBMITS:
    plt.hist(ScoreMeritMean[indices], label=f'Resubmit, $N_r$={len(indices)}', alpha=0.5, bins=bins)
    plt.legend()

plt.title(f'SSW14-19, N={num_proposals}')
plt.ylabel('N')
plt.xlabel('Merit Mean')
plt.xticks([1,2,3,4,5])

plt.tight_layout()

m = np.nanmean(ScoreMeritMean)
m_r = np.nanmean(ScoreMeritMean[indices])
frac = len(indices) / num_proposals

print(f'SSW{np.amin(Year)}-{np.amax(Year)}: N={num_proposals:3}  N_r={len(indices):3} ({frac*100:2.0f}%)' +
      f'  Mean={m:5.3}  Mean_r={m_r:5.3}')
plt.show()
            
            