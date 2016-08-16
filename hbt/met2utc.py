# MET2UTC.py -- Convert from MET to UTC, for New Horizons only.
#
# Returns error message for any other mission.
#
# Assumes that the SPICE Leap seconds kernel and New Horizons
# spacecraft clock (SCLK) kernel have already been loaded. 

# The NH sclk produces a "tick" every 20 microseconds, so there are
# 50,000 ticks in 1 second of MET
#
# Note, 1 "second" of MET is not quite equal to 1 second of time, since the
# spacecraft clock drifts a tiny amount. 
#
# A. Steffl, Mar 2008  -- Original version
# H. Throop, Apr 2008  -- Modified for GV
# H. Throop, Apr 2013  -- Changed sclk_ticks from double to float, as per Nathaniel Cunningham
# H. Throop, Aug 2016  -- Converted to python

import hbt
import numpy as np
import cspice

def met2utc(met_in, name_observer = 'NEW HORIZONS'):

#     met_in = 299348928.9358144
#     name_observer = 'New Horizons'

  if (name_observer.upper().find('NEW HORIZONS') == -1):
    print 'MET can be used only for New Horizons'
    return

# Convert input to an array, even if it is not

  if hbt.is_array(met_in):
    met = np.array(met_in)
  else:
    met = np.array([met_in])

# If there are any commas in the MET, then remove them

  if (type(met[0]) == str):
    met = np.zeros(np.shape(met_in))
    for i,met_i in enumerate(met_in):
      met[i] = float(met_in[i].replace(',', ''))

  sclk_ticks = np.array(met * 5e4)  # Have to include np.array() -- not sure why. I guess a 1x1 np-array is demoted??
  ntime = np.size(met_in)     # Number of elements
  et  = np.zeros(ntime) 
  utc = np.zeros(ntime, dtype = 'S30')

  for i in range(np.size(ntime)):  
     et[i] = cspice.sct2e(-98, sclk_ticks[i])
     utc[i] = cspice.et2utc(et[i], 'C', 3)
#        utc[i] = cspice_et2utc, et_i, 'ISOD', 3, utc_i
  
  if (ntime == 1):
    utc = utc[0]

  return utc
