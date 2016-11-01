# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:36:37 2016

@author: throop
"""


import pdb
import glob
import math       # We use this to get pi. Documentation says math is 'always available' 
                  # but apparently it still must be imported.
from   subprocess import call
import warnings
import pdb
import os.path
import os
import subprocess

import astropy
from   astropy.io import fits
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot
from   matplotlib.figure import Figure
import numpy as np
import astropy.modeling
from   scipy.optimize import curve_fit
#from   pylab import *  # So I can change plot size.
                       # Pylab defines the 'plot' command
#import cspice
#from   itertools import izip    # To loop over groups in a table -- see astropy tables docs
#from   astropy.wcs import WCS
#from   astropy.vo.client import conesearch # Virtual Observatory, ie star catalogs
from   astropy import units as u           # Units library
from   astropy.coordinates import SkyCoord # To define coordinates to use in star search
#from   photutils import datasets
from   scipy.stats import mode
from   scipy.stats import linregress
#import wcsaxes
import time
from scipy.interpolate import griddata

import re # Regexp
import pickle # For load/save

import cProfile # For profiling

# Imports for Tk

#import Tkinter
#import ttk
#import tkMessageBox
#from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from   matplotlib.figure import Figure

from astropy.utils.misc import set_locale

# HBT imports

#import hbt


# Process Uwingu sales data

# Initialize the filenames

def parse_coupon(s):
    """
    Parse "code:vip|description:VIP Valentines Day Code|amount:10.00" into a 3-tuple of (code, description, amount)
    """
    
    first = s.split(';code')[0]  # Remove any coupons after the 1st one. We ignore them entirely.
    print(s)
    parts = first.split('|')
    
    out = (parts[0].split(':')[1],
           parts[1].split(':')[1],
           parts[2].split(':')[1])  
    
    return out

DO_READ_PICKLE = True
    
dir_data = '/Users/throop/Data/Uwingu/SalesCraters'
#files    = glob.glob(dir_data + '/*orders*.csv')
file_csv  = 'uwingu-orders-export_all.csv'
file_pickle = file_csv.replace('.csv', '.pkl')
file_coupons = 'Uwingu03-Oct-2016_15_15_06 Coupon History.csv'

if (DO_READ_PICKLE):
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()
    print('Read: ' + file_pickle)

else:            

    # Read the main table from csv (which is very very slow)
    
    with set_locale('en_US.UTF-8'):
        t = Table.read(dir_data + '/' + 'uwingu-orders-export_all.csv', format='ascii') # This is really slow. Several minutes to read 23K-line file.

# Sort it in place (overwrites existing)

t.sort(keys='order_id')

# Read the coupons table

tc = Table.read(dir_data + '/' + file_coupons, format='ascii.csv')


# Add new columns as needed
mpl.rc('lines', linestyle='none', marker = '.')

# Notes:
# There is one product per line. If someone bought two craters, it is two lines.
# In the 'coupons' field, this is set the same on every item in the transaction.
#  e.g., if two craters with one code, then 'coupons' field lists the coupon code
# for both the first crater, and the second.
#
# If multiple coupons are used, then they are all listed in the 'coupons' field, appended with ';'.
# 
# 'order_total' -- this is the post-coupon amount. For all of Ian Harnett's purchases, this is zero.
# 'item_total'  -- this is the raw price of the item, pre-coupon.
# 'order_id'    -- If someone orders multiple items, all of the rows have the same order ID.
#
# Printed certs: Has item_meta = "Certificate Format=Downloadable PLUS Printed &amp= Framed Certificate (&#036=39.95)" and shipping_items = "method:'
#                They also have a 'shipping_total'.
#                For these, item_total is the crater + $39.95 cert.
#                item_total does *not* include shipping.
#
# CMCFM: These have the crater cost embedded in the 'item_meta'. Regular craters do not.
#
# Cert design: These are embedded in 'item_meta' for CMCFM, but not regular craters.
#               I'm not sure there is any way to get them for regular craters.

# Extract arrays to mask each type of sale
# I'm sure there's a slicker way to do this, but I don't know what it is.

num_rows     = len(t)

is_crater    = np.zeros(num_rows, dtype='bool')
is_vote      = np.zeros(num_rows, dtype='bool')
is_gift_cert = np.zeros(num_rows, dtype='bool') 
is_udse      = np.zeros(num_rows, dtype='bool')
is_cmcfm     = np.zeros(num_rows, dtype='bool')
is_udse_gift = np.zeros(num_rows, dtype='bool')
is_planet    = np.zeros(num_rows, dtype='bool')
is_ppd       = np.zeros(num_rows, dtype='bool')
is_bm2m      = np.zeros(num_rows, dtype='bool')
coupon_code  = np.zeros(num_rows, dtype='S100')
coupon_amount= np.zeros(num_rows, dtype='f')
coupon_description= np.zeros(num_rows, dtype='S100') # This is not a string, but a bytelist or something dfft.
has_framed_cert = np.zeros(num_rows, dtype='bool')
num_items_this_order = np.zeros(num_rows, dtype='int')
is_coupon1_promo     = np.zeros(num_rows, dtype='bool')
is_coupon1_gift_cert = np.zeros(num_rows, dtype='bool')

for i in range(len(t)):    
    item = t[i]['item_name']
    coupon = t[i]['coupons']
    meta   = t[i]['item_meta']
    ordernum=t[i]['order_number']
 
    is_vote[i]   = (item[0:11]    == 'Planet Vote')
    is_crater[i] = (item[0:11]    == 'Mars Crater')
    is_gift_cert[i] = (item[0:11] == 'Gift Certif')
    is_cmcfm[i] = (item[0:11]     == 'Choose My C')
    is_udse[i] = (item[0:11]      == 'Uwingu Dail')
    is_planet[i] = (item[0:11]    == 'Planet Name')
    is_udse_gift[i] = (item[0:11] == 'Gift Subscr')
    is_ppd[i] = (item[0:11]       == 'Mars Provin')
    is_bm2m[i] = (item[0:11]      == 'Beam Me To ')
    
    has_framed_cert[i] = ('Framed Certificate' in meta)
    num_items_this_order[i] = np.sum(t['order_number'] == ordernum)
    
    # Break the 'coupons' field into three new columns
    
    if (repr(type(coupon)) != "<class 'numpy.ma.core.MaskedConstant'>"):
        (coupon_code[i], coupon_description[i], coupon_amount[i]) = parse_coupon(coupon)

t['coupon_code'] = coupon_code
t['coupon_description'] = coupon_description
t['coupon_amount'] = coupon_amount
t['has_framed_cert'] = has_framed_cert # Boolean
    
# Write the main sales table to a pickle file

lun = open(dir_data + '/' + file_pickle, 'wb')
pickle.dump(t, lun)
lun.close()
print('Wrote: ' + file_save)
        
is_other = (is_vote + is_crater + is_gift_cert + is_cmcfm + is_udse + is_planet + is_udse_gift + is_ppd + is_bm2m) == 0    
    
# Make a histogram of craters

# Remove the non-craters

# Make a list of all of the promo coupons. I assmbled this manually from 
list_coupons_promo = {'astronomy today',
                      'astronomytoday',
                      'carter',
                      'dad2016',
                      'gift2015',
                      'happy holidays',
                      'happyholidays',
                      'holiday2015',
                      'mars one',
                      'marsone',
                      'uwinguvip',
                      'valentine',
                      'vip'}
                      
h = np.histogram(t[is_crater]['item_total'], bins=5000)
price = h[1][0:-1]
num   = h[0]
w = (num != 0)

price = price[w]
num   = num[w]
for i in range(np.size(price)):
    print("{}  ${}  {}".format(i, price[i], num[i]))

#==============================================================================
# Print the unique coupon codes
#==============================================================================

cc = sorted(np.unique(coupon_code))
for cci in cc:
    print((cci) + ' ' + repr(np.sum(t['coupon_code'] == cci)))

plt.plot(price, num)
plt.y
#item_total
#order_total
#refunded_total -- have to ignore any rows with this in it

# Get a list of all the unique coupons
# A few coupon codes to note:
#    'astronomy today'
#    'astronomytoday'
#    'astronomymag'
#    'dad2016'
#    'gift2015'
#    'happy holidays'
#    'happyholidays'
#    'holiday2015'
#    'marsone'  # NB:Amount of this is $10, $9.99, $8.91, $5 -- all of these!
#    'mars one'

# Add several more columns:
#   is_coupon_promo
#   is_coupon_gift_cert
#   num_items_this_order
#    

#==============================================================================
#  Make Some Plots
#==============================================================================

#Make a 


# My goal: Make a plot of sales statistics: dollar amount vs. #. In this case the dollar amount is the raw crater price itself.
#
# Then, make the same plot, but using not the intrinsic price of the crater, but using the *discounted* price (ie, any coupon that has been applied).
# 