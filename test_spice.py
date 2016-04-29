# HBT 15-Jun-2015

import cspice

print "Spice imported"
cspice.furnsh("/Users/throop/gv/dev/gv_kernels_new_horizons.txt")

UT = "2015 1 Jun 00:00:00"

et = cspice.utc2et(UT)			 # Works OK

print "ET = " + repr(et)

utc = cspice.et2utc(et, "C", 1)		 # Crashes the python kernel
					 # Works now after recompiling the kernel!

print "UTC = " + utc


