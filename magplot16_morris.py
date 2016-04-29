#! \Users\Daniel\Anaconda\python
# Modified by HBT 25-Jan-2015

from scipy.optimize import curve_fit
import numpy as np

# Define sinusoid function, with Amplitude D, Frequency A, phase B, DC offset C
# I added parameter D, which was missing.

def f(x, a, b, c, d):  
	return(d * np.sin(a*x+b)+c)  
 
import matplotlib.pyplot as plt
n=0; x=[]; y=[]; yerror=[]
data=open('magdata.txt')

for line in data: 
		
	n+=1
	splits=line.split()
	x.append(float(splits[7]))
	y.append(float(splits[1]))
	yerror.append(float(splits[8]))

# curve_fit requires some initial guesses, especially for a large-dimensional fit
# like this. By eye-balling it, I came up with a good initial guess, which is plotted below.

a0 = 0.0065  # Frequency
b0 = 0     # x offset (ie, phase)
c0 = 19.3   # y offset (dc bias)
d0 = 0.6   # scaling

# Take all of the initial guesses and stuff them into an array to pass to curve_fit

p0 = np.array([a0, b0, c0, d0])

# Call the curve fit routine

(popt, pcov) = curve_fit(f, x, y, p0=p0, sigma=yerror, maxfev=1500)

# Extract the fit results

(aopt, bopt, copt, dopt) = popt
maxx = max(x)
minx = min(x)
ref = np.arange(minx, maxx, (maxx-minx)/n)

# Convert x into a numpy array to plot it easily

x = np.array(x)

# Compute the function for the initial guesses

y0 = f(x, a0, b0, c0, d0)

plt.errorbar(x,y,yerror, ls='None',marker='.', label = 'Initial Guess')
plt.plot(x,y0, color='green', label = 'Initial Guess')

# Plot the function for the solved values

plt.scatter(ref, f(ref,aopt,bopt,copt,dopt), color='r',marker='.', label='Fit')

# Tweak the X and Y axes slightly

plt.ylim((18,22))
plt.xlim((2500,12500))
plt.legend()
plt.show()