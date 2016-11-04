# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:36:37 2016

@author: throop
"""


import pdb
import glob

import pdb

import scipy.stats

import astropy
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import matplotlib.pyplot as plt # pyplot
from   matplotlib.figure import Figure
import numpy as np
import astropy.modeling
import astropy.time as Time

#from   photutils import datasets
#import wcsaxes
import time

import pickle # For load/save


# Imports for Tk

#import Tkinter
#import ttk
#import tkMessageBox
#from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from astropy.utils.misc import set_locale

# HBT imports

#import hbt


# Process Uwingu sales data

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
# Printed certs: Has item_meta = "Certificate Format=Downloadable PLUS Printed &am
#                 Framed Certificate (&#036=39.95)" and shipping_items = "method:'
#                They also have a 'shipping_total'.
#                For these, item_total is the crater + $39.95 cert.
#                item_total does *not* include shipping.
#
# CMCFM: These have the crater cost embedded in the 'item_meta'. Regular craters do not.
#
# Cert design: These are embedded in 'item_meta' for CMCFM, but not regular craters.
#               I'm not sure there is any way to get them for regular craters.



#==============================================================================
# Parse a coupon string into its components
#==============================================================================

def parse_coupon(s):
    """
    Parse "code:vip|description:VIP Valentines Day Code|amount:10.00" into a 3-tuple of (code, description, amount)
    """
    
    first = s.split(';code')[0]  # Remove any coupons after the 1st one. We ignore them entirely.
#    print(s)
    parts = first.split('|')
    
    out = (parts[0].split(':')[1],
           parts[1].split(':')[1],
           parts[2].split(':')[1])  
    
    return out

#==============================================================================
# Display a table of info about an item/s
#==============================================================================

def display_item(t,arr):
  t[arr]['order_total', 'shipping_total', 'cart_discount', 'item_name', 
         'item_total', 'item_actual_paid', 'item_total_unframed', 
         'coupon_amount', 'coupon_code', 'has_framed_cert'].pprint(max_width=400, max_lines=100)

#==============================================================================
# Plot histogram of a group of craters
#==============================================================================

def plot_hist_craters(num, bins, title=''):
    plt.hist(num, bins, facecolor='yellow')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dollars')
    plt.ylabel('Number Sold')
    plt.title(title)
    plt.show()

#==============================================================================
# Plot new users (and total users) per month
#==============================================================================

def plot_monthly_users(t):
    
    jd = t['jd']
    jd_min = np.amin(jd)
    jd_max = np.amax(jd)
    jd_range = jd_max - jd_min
    
    dt = 30  # Plot by month
    num_bins = int(jd_range / dt)  # 23 bins total, or whatever
    d_bin = jd_range / (num_bins-1) # Width of each bin, in days
    bins = np.arange(jd_min, jd_max, d_bin) # Might be an off-by-one issue here

    num_users = 0 * bins.copy()
    
    # Go thru and manually search for the number of unique users
    # in month 0..1, then 0..2, then 0..3, etc.
    # This is brute force and slow, but it works.
    
    # Cannot just search for the max userid -- that is 30K, but we have only 10K 
    # who have actually bought something.
    
    for i in range(num_bins-2):
        is_good_jd = t['jd'] < bins[i + 1]
        num_users[i] = np.size(np.unique(t[is_good_jd]['customer_id']))
    
    added = num_users - np.roll(num_users,1)
    num_users = num_users[0:-2]
    added     = added[0:-2]
    
    # Get the total # of active users
     
    total = np.size(np.unique(t['customer_id']))
    
    plt.plot(added)
    plt.xlabel('Month')
    plt.ylabel('New Users Added Per Month')
    plt.title('New Active Users, total = ' + repr(total))
    plt.ylim((0,np.amax(added[1:])))
    plt.show()
    
    plt.plot(num_users)
    plt.xlabel('Month')
    plt.ylabel('Total Users')
    plt.title('Active Users, total = ' + repr(total))
    plt.ylim((np.amin(num_users),np.amax(num_users[1:])))
    plt.show()
    
#==============================================================================
# Plot a time series of sales
#==============================================================================

def plot_v_time(t, mask, val='Number', chunk='Month', title='', 
                skip_first=False, show_total=True):

    jd = t['jd']
    jd_min = np.amin(jd)
    jd_max = np.amax(jd)
    jd_range = jd_max - jd_min
    
    times = {'Day':      1, 
             'Week':     7, 
             'Month':   30, 
             'Quarter': 90, 
             'Year' :  365}

    dt = times[chunk]  # Bin with, in days
        
    num_bins = int(jd_range / dt)  # 23 bins total, or whatever
    d_bin = jd_range / (num_bins-1) # Width of each bin, in days
    
    bins = np.arange(jd_min, jd_max, d_bin) # Might be an off-by-one issue here
    
    if (val == 'Number'):
        (number, bins) = \
            np.histogram((t[mask]['jd']-jd_min)/dt, bins=num_bins)
        yval = number
        valstr = ''
        
#        plt.hist((t[mask]['jd']-jd0)/dt, bins=(bins-np.amin(bins))/dt, facecolor='pink')

# For revenue, we want to count how many are in each JD bin, and then 
# sum the dollar values in a corresponding bin. We use 
# binned_statistic() for this. Sort of like IDL's reverse_indices.

    if (val == 'Revenue'):
        (revenue, bins, indices) = \
            scipy.stats.binned_statistic((t[mask]['jd']-jd0)/dt, 
                                          t[mask]['item_total'], 'sum', num_bins)
        yval = revenue
        valstr = '$'
        
# plt.bar() -- makes same plot as histogram
            
    plt.bar(bins[0:-1], yval, facecolor='pink')
    
    total = np.sum(yval)
    
    if (skip_first):
        plt.ylim((0, np.amax(yval[1:])))
    
    plt.xlabel(chunk)
    plt.ylabel(val + ' per ' + chunk)
    if (show_total):    
        plt.title(title + ', total = ' + valstr + repr(int(total)))
    else:
        plt.title(title)
    plt.show()

#==============================================================================
# Start of main program
#==============================================================================

DO_READ_PICKLE = True

PRICE_CERT = 39.95
    
dir_data = '/Users/throop/Data/Uwingu/SalesCraters'
#files    = glob.glob(dir_data + '/*orders*.csv')
file_csv  = 'uwingu-orders-export_all.csv'
file_pickle = file_csv.replace('.csv', '.pkl')
file_coupons = 'Uwingu03-Oct-2016_15_15_06 Coupon History.csv'

if (DO_READ_PICKLE):
    print('Reading pickle file: ' + dir_data + '/' + file_pickle)
    lun = open(dir_data + '/' + file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

else:            

    # Read the main table from csv (which is very very slow)
    
    with set_locale('en_US.UTF-8'):
        print("Reading CSV file " + dir_data + "/" + file_csv)
        t = Table.read(dir_data + '/' + file_csv, format='ascii') 
        # This is really slow. Several minutes to read 23K-line file.
        # Format must be 'ascii', not 'csv', in order to read unicode properly.

# Sort it in place (overwrites existing)

t.sort(keys='order_id')

# Read the coupons table

tc = Table.read(dir_data + '/' + file_coupons, format='ascii.csv')

# Make a list of all of the promo coupons. I assmbled this manually from the list of codes actually used

list_coupons_promo = {'astronomy today',
                      'astronomytoday',
                      'astronomymag',
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

list_ppl_uwu = {'Throop', 'Stern', 'Burton', 'Butler', 'McFarland', 'Tamblyn', 'CoBabe'}

list_ppl_uwu_short = {'alish', 'ellen', 'tomtj', 'aster', 'tjmbt', 'jnoon', 'henry'}
    
# Add new columns as needed

num_rows     = len(t)

is_crater            = np.zeros(num_rows, dtype='bool')
is_vote              = np.zeros(num_rows, dtype='bool')
is_gift_cert         = np.zeros(num_rows, dtype='bool') 
is_udse              = np.zeros(num_rows, dtype='bool')
is_cmcfm             = np.zeros(num_rows, dtype='bool')
is_udse_gift         = np.zeros(num_rows, dtype='bool')
is_planet            = np.zeros(num_rows, dtype='bool')
is_ppd               = np.zeros(num_rows, dtype='bool')
is_bm2m              = np.zeros(num_rows, dtype='bool')
coupon_code          = np.zeros(num_rows, dtype='U100') # U means unicode string. S is a list of bytes, 
                                                        # which in py3 is dfft than a string.
coupon_amount        = np.zeros(num_rows, dtype='f')
coupon_description   = np.zeros(num_rows, dtype='U100')
has_framed_cert      = np.zeros(num_rows, dtype='bool')
num_items_this_order = np.zeros(num_rows, dtype='int')
is_coupon1_promo     = np.zeros(num_rows, dtype='bool')
is_coupon1_gift_cert = np.zeros(num_rows, dtype='bool')
item_total_unframed  = np.zeros(num_rows, dtype='f')    # 'item_total', minus cert framing and shipping
is_test              = np.zeros(num_rows, dtype='bool') # If this is an internal testing purchase
is_completed         = np.zeros(num_rows, dtype='bool')
item_actual_paid     = np.zeros(num_rows, dtype='f')
jd                   = np.zeros(num_rows, dtype='f')    # Julian Date


for i in range(len(t)):    
    item = t[i]['item_name']
    coupon = t[i]['coupons']
    meta   = t[i]['item_meta']
    ordernum=t[i]['order_number']

# Determine what kind of purchase each one was
 
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
    is_completed[i] = (t[i]['status'] == 'completed') # Other status is 'cancelled'
    num_items_this_order[i] = np.sum(t['order_number'] == ordernum)


    is_test[i] = False

#   Check if it was a purchase by one of us. Flag all of these as tests, even if that is not 100% accurate
    
    if t[i]['billing_last_name'] in list_ppl_uwu:
        is_test[i] = True

    # Break the 'coupons' field into three new columns
    
    if (repr(type(coupon)) != "<class 'numpy.ma.core.MaskedConstant'>"):
        (coupon_code[i], coupon_description[i], coupon_amount[i]) = parse_coupon(coupon)

# Check if the coupon code is custom-made for Tom, Ellen, etc.
# If so, flag as a test
        
        if coupon_code[i][0:5] in list_ppl_uwu_short:
            is_test[i] = True

        if coupon_code[i] in list_coupons_promo:
            is_coupon1_promo[i] = True

# Now do the math to figure out the actual amount paid for the crater.
# Subtract off the cert price, and then subtract any discounts applied to the full order.

    item_total_unframed[i] = t[i]['item_total']

    if (has_framed_cert[i]):
        item_total_unframed[i] -= (PRICE_CERT + t[i]['shipping_total'])

# Now set the actual dollar value really paid. 
# It turns out that we have already calculated this. 
# This works for items paid with gift certs, and items with promo amounts applied.
# ** Actually, this is not true. Behavior is different for different coupons.
# e.g., 'item_total' for 'happyholidays' and 'dad2016' are different. One is pre-discount, 
# and one is post. So there is no way to know, except by studying it for each code individually.
# Ugh. 

    item_actual_paid[i] = item_total_unframed[i]

#    if (is_coupon1_promo[i]):
#        item_actual_paid[i] = item_total_unframed[i] + t[i]['order_discount']/num_items_this_order[i]
    

# Convert time to JD
time = astropy.time.Time(t['date'], format='iso', scale='utc')
time.format = 'jd'
jd = time.value
    
# Now that we have calculated all of these new columns, add them to the main table

t['is_crater']       = is_crater
t['is_vote']         = is_vote
t['is_gift_cert']    = is_gift_cert
t['is_udse']         = is_udse
t['is_cmcfm']        = is_cmcfm
t['is_udse_gift']    = is_udse_gift
t['is_planet']       = is_planet
t['is_ppd']          = is_ppd
t['is_bm2m']         = is_bm2m

t['coupon_code']     = coupon_code
t['coupon_description'] = coupon_description
t['coupon_amount']   = coupon_amount
t['has_framed_cert'] = has_framed_cert # Boolean
t['is_completed']    = is_completed
t['is_test']         = is_test 
t['item_total_unframed'] = item_total_unframed
t['item_actual_paid']    = item_actual_paid
t['is_coupon1_promo']    = is_coupon1_promo
t['jd']                  = jd

# Write the main sales table to a pickle file

lun = open(dir_data + '/' + file_pickle, 'wb')
pickle.dump(t, lun)
lun.close()
print('Wrote: ' + file_pickle)
        
is_other = (is_vote + is_crater + is_gift_cert + is_cmcfm + is_udse + \
            is_planet + is_udse_gift + is_ppd + is_bm2m) == 0    
    
# Make a histogram of craters

# Remove the non-craters



#==============================================================================
# Extract a table for just the craters
#==============================================================================

t_crater = t[is_crater]
                    
#==============================================================================
# Make some histograms
#==============================================================================

price = t_crater['item_total']
price_max = int(np.amax(price))
                   
(num_out, bins_out) = np.histogram(price, bins=price_max)

# Make a scatter plot of the number of sales. These are in $1-wide bins.

plt.plot(bins_out[0:-1], num_out)
plt.plot(0.9, num_out[0])  # Plot the zero-dollar value. It can't be on log-log
                           # so we just put it a bit to the left of 1.0 .
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Number of Sales')
plt.xlabel('Price [$], after discount')
plt.xlim((0.7, np.amax(bins_out)))
plt.show()

# Examine one of these in detail. e.g., where are there 9 in the $212 range?
# A: That is the actual price. It must be a $250 crater that we were 
# discounting by 15%.

w = np.logical_and((t_crater['item_total'] >= 212), (t_crater['item_total']<213))
display_item(t_crater, w)

# Makea plot of just items with framed certs

w = (t_crater['has_framed_cert'] == True)
display_item(t_crater, w)

#==============================================================================
# List sales statistics to a table
#==============================================================================

#price = price[w]
#num   = num[w]

for i in range(np.size(bins_out)-2):
    if (num_out[i] != 0):
        print("{:>5}  ${:>8} .. ${:>8}  {:>5}".format(i, bins_out[i], \
              bins_out[i+1]-0.01, num_out[i]))

    
#==============================================================================
# Print the unique coupon codes, if requested
#==============================================================================

DO_LIST_COUPON_CODES = False

cc = sorted(np.unique(coupon_code))

if DO_LIST_COUPON_CODES:
    for cci in cc:
        print((cci) + ' ' + repr(np.sum(t['coupon_code'] == cci)))

#==============================================================================
# Make some plots -- HISTOGRAMS
#==============================================================================

# Define some good bin limits for crater prices

bins_price_crater = np.array([0,1,6,11,26,51,101,251,501,1001,2501, 5001])

is_good = np.logical_and(t['is_completed'], np.logical_not(t['is_test']))

# Select all regular craters

w = np.logical_and(is_good, t['is_crater'])
plot_hist_craters(t['item_actual_paid'][w], bins_price_crater, 'All Craters, N = ' + repr(np.sum(w)))

# CMCFM only

w = np.logical_and(is_good, t['is_cmcfm'])
plot_hist_craters(t['item_actual_paid'][w], bins_price_crater, 'CMCFM, N = ' + repr(np.sum(w)))

# Ian Harnett only

w = np.logical_and(is_good,
                   t['billing_last_name'] == 'Harnett')
plot_hist_craters(t['item_actual_paid'][w], bins_price_crater, 'Harnett, N = ' + repr(np.sum(w)))

# Framed certs only

w = np.logical_and(is_good,
                   t['has_framed_cert'])
plot_hist_craters(t['item_actual_paid'][w], bins_price_crater, 'Framed Cert, N = ' + repr(np.sum(w)))

#==============================================================================
# Make some plots - TIME SERIES
#==============================================================================

jd0 = jd[0]

num_weeks = int( (np.amax(jd) - np.amin(jd))/7 )

w = np.logical_and(is_good, t['is_vote'])
#plot_weekly(t[w]['jd'], num_weeks, 'Votes')

w = np.logical_and(is_good, t['is_planet'])
#plot_weekly(t[w]['jd'], num_weeks, 'Planet Name')

w = np.logical_and(is_good, t['is_cmcfm'])
#plot_weekly(t[w]['jd'], num_weeks, 'CMCFM')
plot_v_time(t, w, val='Number', chunk='Month', title = 'CMCFM', skip_first=True)
plot_v_time(t, w, val='Revenue', chunk='Month', title = 'CMCFM', skip_first=True)

w = np.logical_and(is_good, t['is_crater'])
#plot_weekly(t[w]['jd'], num_weeks, 'Craters')
plot_v_time(t, w, val='Revenue', chunk='Month', title = 'Craters')

w = np.logical_and(is_good, t['billing_last_name'] == 'Harnett')
plot_v_time(t, w, val='Number', chunk='Week', title = 'Harnett', skip_first=True)
plot_v_time(t, w, val='Revenue', chunk='Month', title = 'Harnett', skip_first=True)

w = np.logical_and(is_good, t['has_framed_cert'])
plot_v_time(t, w, val='Number', chunk='Month', title = 'Framed Certs', show_total=True, skip_first=False)

w = np.logical_and(is_good, t['is_bm2m'])
plot_v_time(t, w, val='Revenue', chunk='Day', title = 'BM2M', skip_first=False)
plot_v_time(t, w, val='Number', chunk='Day', title = 'BM2M', skip_first=False)

w = np.logical_and(is_good, t['is_udse'])
#plot_weekly(t[w]['jd'], num_weeks, 'UDSE')
plot_v_time(t, w, val='Number', chunk='Month', title = 'UDSE', skip_first=False)

w = np.logical_and(is_good, t['is_gift_cert'])
plot_v_time(t, w, val='Number', chunk='Month', title = 'Gift Cert', skip_first=False)
plot_v_time(t, w, val='Revenue', chunk='Month', title = 'Gift Cert', skip_first=False)

#plot_weekly(t[w]['jd'], num_weeks, 'Gift Cert')

plot_v_time(t, w, val='Number', chunk='Week', title = 'Gift Cert')
plot_v_time(t, w, val='Number', chunk='Day', title = 'Gift Cert')

#==============================================================================
# Plot new users (and total users) per month
#==============================================================================

plot_monthly_users(t)

    
    ### XXX There is some problem here. I am assuming that customer_id=30K
    # implies we have 30K customers, and that is not true. There are a lot
    # of skipped values.

#==============================================================================
# Plot monthly revenue of each stream
#==============================================================================

jd = t['jd']
jd_min = np.amin(jd)
jd_max = np.amax(jd)
jd_range = jd_max - jd_min

dt = 30  # Plot by month
num_bins = int(jd_range / dt)  # 23 bins total, or whatever
d_bin = jd_range / (num_bins-1) # Width of each bin, in days
bins = np.arange(jd_min, jd_max, d_bin) # Might be an off-by-one issue here
    
w = np.logical_and(is_good, t['is_crater'])
(revenue_craters, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)

w = np.logical_and(is_good, t['is_cmcfm'])
(revenue_cmcfm, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)

w = np.logical_and(is_good, t['is_bm2m'])
(revenue_bm2m, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)

w = np.logical_and(is_good, t['is_vote'])
(revenue_vote, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)

w = np.logical_and(is_good, t['is_planet'])
(revenue_planet, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)
    
w = np.logical_and(is_good, t['is_udse'])
(revenue_udse, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)
### XXX There is some error here in excluding Harnett.

w = np.logical_and(is_good, 
                   np.logical_and(t['is_gift_cert'], t['shipping_last_name'] != 'Harnett'))
(revenue_gift_cert, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  t[w]['item_total'], 'sum', bins)


w = np.logical_and(is_good, t['has_framed_cert'])
(revenue_framed, bins_out, indices) = \
    scipy.stats.binned_statistic((t[w]['jd']), 
                                  PRICE_CERT+np.zeros(np.sum(w)), 'sum', bins)    
    
revenue_total = revenue_craters + revenue_cmcfm + revenue_bm2m + revenue_vote + \
                revenue_planet + revenue_gift_cert + revenue_udse + revenue_framed

revenue_craters_frac = revenue_craters / revenue_total
revenue_cmcfm_frac        = revenue_cmcfm / revenue_total
revenue_bm2m_frac         = revenue_bm2m / revenue_total
revenue_vote_frac         = revenue_vote / revenue_total
revenue_planet_frac       = revenue_planet / revenue_total
revenue_gift_cert_frac    = revenue_gift_cert / revenue_total
revenue_udse_frac = revenue_udse / revenue_total
revenue_framed_frac      = revenue_framed / revenue_total


#hbt.figsize((15,10))

plt.stackplot((bins[0:-1]-np.amin(bins))/dt, \
               revenue_vote_frac, 
               revenue_planet_frac,
               revenue_udse_frac,
               revenue_framed_frac, 
               revenue_bm2m_frac, 
               revenue_craters_frac, 
               revenue_gift_cert_frac, 
               revenue_cmcfm_frac, 

               colors = 
               ['pink', 'lightgreen', 'lightblue', 
                         'yellow', 'salmon', 'lightgrey', 'cornsilk', 'aqua'],
               labels=
               ['Vote', 'Planet', 'UDSE', 
                         'Framed', 'BM2M', 'Craters', 'GC', 'CMCFM'])
 
plt.xlim((0,(np.amax(bins[0:-1])-np.amin(bins))/dt))
plt.ylim((0,1))
plt.legend()
plt.ylabel('Fraction')
plt.xlabel('Month')
plt.title('Revenue streams, Fraction of Total')
plt.show()

#==============================================================================
# Plot gift certs as fraction of total craters
#==============================================================================

w = np.logical_and(is_good, t['is_crater'])
(num_craters,bins_out) = np.histogram(t[w]['item_total_unframed'], bins=bins_price_crater)

w = np.logical_and(is_good, t['has_framed_cert'])
(num_certs,bins_out) = np.histogram(t[w]['item_total_unframed'], bins=bins_price_crater)

# XXX I'm not sure this is totally right. I might be confusing raw price
# with framed price, something like that. But I think it gives the right general picture.

plt.plot(bins_price_crater[0:-1], num_certs / num_craters * 100)
plt.ylabel('Percent Framed')
plt.xlabel('Crater Price [$]')
plt.xscale('log')
plt.title('Framing as Percent of Total Crater Sales')
plt.show()

#==============================================================================
# Make some more plots
#==============================================================================

# Total monthly revenue, from all sources summed
# % makeup of total sales: craters vs. UDSE, etc.
# % of orders with framing
# Num of users vs. time (or, new users per chunk) [DONE]

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


# My goal: Make a plot of sales statistics: dollar amount vs. #. In this case 
# the dollar amount is the raw crater price itself.
#
# Then, make the same plot, but using not the intrinsic price of the crater, but using 
# the *discounted* price (ie, any coupon that has been applied).
# 