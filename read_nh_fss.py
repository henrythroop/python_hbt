import pdb
import sys # for sys.stdout.write

from   subprocess import call
import string
import glob
import os       # for chdir()
import os.path  # for isfile()
# import astropy
# from   astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt # pyplot 
from   subprocess import call   # Shell output
from   subprocess import check_output   # Shell output

import numpy as np
from   pandas import DataFrame, Series
from   pylab import *
from   astropy.convolution import convolve, Box1DKernel
import cspice

file_in = "/Users/throop/Data/NH_FSS/ss_tlm_299182134_299194367.csv"

# From FSS manual:
# FOV: 19 deg
# Accuracy: 0.05 deg
# Resolution: 0.004 deg

# Four columns: MET, SCUTC, Data_good, Angle_rad, Angle_deg

# Read in the file

print "Loading file " + file_in
d = loadtxt(file_in, delimiter = ',', skiprows=1, dtype = 'S100')
#   l = len(file)

# Convert from list into NP array

met = np.array(d[:,0], dtype = 'f16')  # 299182135.00, etc. dtype 'f' is insufficient.
scutc = np.array(d[:,1]) # '2015-195 // 12:16:56.875804', etc
data_good = np.array(d[:,2], dtype = 'b') # boolean flag
angle_rad = np.array(d[:,3], dtype = 'f')
angle_deg = np.array(d[:,4], dtype = 'f')

# Zoom into the first one

plot(met, angle_deg)
plt.xlim((2.9918e8 + 3500, 2.9918e8 + 5000))

plot(met, data_good, marker = '+', linestyle = 'none')

# Find segments where the 'data_good' flag are false

dpocc = 3400
dpocc = 3500
dcocc = 1000

met_pocc = 2.9918e8 + 3300	# First occultation. Call it p_occ (though I'm not sure)

met_cocc = 2.9918e8 + 9000	# Second occultation. Call it c_occ.

# Plot the Pluto ingress


index_pocc = where(met > met_pocc)[0][0]

t = met - met[index_pocc]

p0 = index_pocc

y_data_good = 2.72

rcParams['figure.figsize'] = 10, 12
plot(t, angle_deg, marker = '+', markersize = 1, linestyle = 'none')
plot(t, data_good * y_data_good, marker = '+', markersize = 1, linestyle = 'none')
plt.xlim((0, 600))
plt.ylim ((2.71, 2.745))
plt.xlabel('Seconds past ' + scutc[p0])
plt.ylabel('Solar angle [deg]')
plt.title('Pluto Ingress')
plt.show()

rcParams['figure.figsize'] = 10, 12
plot(t, angle_deg, marker = '+', markersize = 1, linestyle = 'none')
plot(t, data_good * y_data_good, marker = '+', markersize = 1, linestyle = 'none')
plt.xlim((400, 500))
plt.ylim ((2.71, 2.745))
plt.xlabel('Seconds past ' + scutc[p0])
plt.ylabel('Solar angle [deg]')
plt.title('Pluto Ingress')
plt.show()

rcParams['figure.figsize'] = 10, 12
plot(t, angle_deg, marker = '+', markersize = 1, linestyle = 'none')
plot(t, data_good * y_data_good, marker = '+', markersize = 1, linestyle = 'none')
plt.xlim((1000,1500))
plt.ylim ((2.70, 2.75))
plt.xlabel('Seconds past ' + scutc[p0])
plt.ylabel('Solar angle [deg]')
plt.title('Pluto Egress')
plt.show()

rcParams['figure.figsize'] = 10, 12
plot(t, angle_deg, marker = '+', markersize = 1, linestyle = 'none')
plot(t, data_good * y_data_good, marker = '+', markersize = 1, linestyle = 'none')
plt.xlim((1100,1200))
plt.ylim ((2.70, 2.75))
plt.xlabel('Seconds past ' + scutc[p0])
plt.ylabel('Solar angle [deg]')
plt.title('Pluto Egress')
plt.show()


# Now we want to get the angular separation between Pluto and the the Sun, at each MET.

# Get Pluto position
# Get Sun position
# Get Pluto radius (actually, use mission-derived value) -> angular radius
# Get Sun radius -> angular radius

cspice.furnsh("/Users/throop/gv/dev/gv_kernels_new_horizons.txt")
et = cspice.utc2et("1 jan 2002 11:11:11")
utc = cspice.et2utc(et, "C", 3)

et = np.copy(met)
for i in range(size(met)/2):
  et[i] = cspice.utc2et(scutc[i])
  print 'converted et[' + repr(i) + ']'

quit

name_obs = 'New Horizons'
frame = 'J2000'
abcorr = 'LT'

# Now call SPKEZR. But warning -- SPICE dies here??

print "Calling SPKEZR Pluto"

(state_plu, ltime) = cspice.spkezr('Pluto', et[0], frame, abcorr, name_obs)
print "Calling SPKEZR SUn"

(state_sun, ltime) = cspice.spkezr('Sun',   et[0], frame, abcorr, name_obs)

# Now call SPKEZR. But warning -- SPICE dies here??

