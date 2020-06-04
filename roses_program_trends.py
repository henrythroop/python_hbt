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

#import hbt

import re

def has_revote(record, column_min, column_max):
    """    
    Takes a record, and sees if there is a revote in it or not.
    """
    s = record[column_min : column_max]
    convert_first_to_generator = (str(item) for item in s)

    s2 = ' '.join(convert_first_to_generator)
    
    # Look for a paren, such as in "4.25 (4.57)", which indicates a revote
    
    result = ('(' in s2) or (')' in s2)
    
    return result

def extract_vote_original(record, column):
    """
    Extract the original vote from a string like "4.00" or "4.25 (4.57)".
    """
    out = str(record[column]).split(' ')[0] 
    
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

ScoreMeritMean = []
ScoreMeritMedian = []

ScoreCostMean = []
ScoreCostMedian = []

ScoreRelevanceMean = []
ScoreRelevanceMedian = []

ScorePMEMean = []
ScorePMEMedian = []

column_NameSubpanel    = 0
column_NumberWeek      = 1
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
NameInstitution = np.array(NameInstitution)
NameSubpanel = np.array(NameSubpanel)
TitleProposal = np.array(TitleProposal)
NumberProposal = np.array(NumberProposal)

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

# Loop through and calculate similarity for every pair of proposals

file_pkl = f'{name_program}_{num_proposals}.pkl'

if os.path.exists(file_pkl):
     (SimilarTitle, SimilarityPI) = pickle.load(open(file_pkl, "rb"))
     print(f'Loaded: {file_pkl}')
else:     
     
    for i in range(num_proposals):
        for j in range(num_proposals):
            SimilarityTitle[i,j] = fuzz.ratio(TitleProposal[i], TitleProposal[j])
            SimilarityPI[i,j]    = fuzz.ratio(NamePI[i],    NamePI[j])
            
        print(f'Done with {i}/{num_proposals}')
    print(f'Saving: {file_pkl}')
    pickle.dump( (SimilarityTitle, SimilarityPI), open( file_pkl, "wb" ) )

CriteriaSimilarity = 75  # This is kind of arbitrary. 75 seems good.

# Now go through and search for matches.
# We end up with a list called 'matches', which indicates each proposal which matches the indicated one.
# Reversals are excluded (ie, if i == j, then j == i is excluded).

matches = []

for i in range(num_proposals):

    matches_i = [] 
    for j in range(num_proposals):
        if j > i:
            if (SimilarityTitle[i,j] > CriteriaSimilarity) and (SimilarityPI[i,j] > CriteriaSimilarity):
                matches_i.append(j)        # Add the j element to list (ie, the 2nd, 3rd, 4th etc proposal)
                if len(matches_i) == 1:
                    matches_i.insert(0,i)  # Add the i element to list (ie, the 1st proposal)
                    print()
                    print("Matched!")
                print(f'{NumberProposal[j]} {NamePI[j]} {TitleProposal[j]}')
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

for i in range(num_proposals):
    m = matches[i]
    if len(m) > 0:
        print(f'{i}')
        for j in range(len(m)):
            print(f'{NumberProposal[m[j]]:15} / {NamePI[m[j]]:25} / {ScoreMeritMean[m[j]]:5} /   {TitleProposal[m[j]]}')
        print()
        
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
plt.title('SSW Merit Mean, 2014-2018')
plt.plot([1,5],[1,5])
plt.gca().set_aspect('equal')
plt.show()

# See what the change would be if we just assigned Merit scores at random!
# This will change every time we run the code.

num_pairs = len(ScoreMeritMean_Y1)

ScoreMeritMean_Y1_s = np.array(random.choices(ScoreMeritMean, k=num_pairs))[0:-1]
ScoreMeritMean_Y2_s = np.array(random.choices(ScoreMeritMean, k=num_pairs))[0:-1]
delta_s = ScoreMeritMean_Y1_s - ScoreMeritMean_Y2_s
plt.hist(delta_s, bins=20)
plt.xlabel('Change in Synthetic Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW2014-2018. Delta mean = {np.nanmean(delta_s):4.2}; Stdev = {np.nanstd(delta_s):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.show()  

# Plot a histogram of the change

delta = ScoreMeritMean_Y2 - ScoreMeritMean_Y1
plt.hist(delta, bins=20, label = 'Actual')
plt.hist(delta_s, bins=20, alpha=0.6, color='pink', label = 'Random')
plt.xlabel('Change in Mean Merit')
plt.ylabel('Number')
plt.title(f'SSW2014-2018. Delta mean = {np.mean(delta):4.2}; Stdev = {np.nanstd(delta):4.2}; N={num_pairs} resubmits')  
plt.axvline(0, color='red', alpha=0.2)
plt.legend()
plt.show()    

# See what the most common institutions are

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


