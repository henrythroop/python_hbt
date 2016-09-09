#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:52:17 2016

@author: throop
"""

##########
# Calc offset between two sets of points. Generates an image for each one, and then calcs relative image shift.
##########

def calc_offset_points(points_1, points_2, shape, plot=False):
    "Calculate the offset between a pair of ordered points -- e.g., an xy list"
    "of star positions, and and xy list of model postns."
    "Returned offset is integer pixels as tuple (dy, dx)."
    
    diam_kernel = 5 # Set the value of the fake stellar image to plot
                    # 5 is best for LORRI. If this is 11, that is too big, and we gt the wrong answer. Very sensitive.

    image_1 = hbt.image_from_list_points(points_1, shape, diam_kernel)
    image_2 = hbt.image_from_list_points(points_2, shape, diam_kernel)
 
    t0,t1 = ird.translation(image_1, image_2) # Return shift, with t0 = (dy, dx). t1 is a flag or quality or something.
    (dy,dx) = t0
    
    if (plot):

        xrange = (0, shape[0]) # Set xlim (aka xrange) s.t. 
        yrange = (shape[1], 0)

        figs = plt.figure()
        ax1 = figs.add_subplot(1,2,1) # nrows, ncols, plotnum. Returns an 'axis'
        ax1.set_aspect('equal') # Need to explicitly set aspect ratio here, otherwise in a multi-plot, it will be rectangular

#        fig1 = plt.imshow(np.log(image_1))
        plt.plot(points_1[:,0], points_1[:,1], marker='o', color='lightgreen', markersize=4, ls='None', label = 'Photometric')
        plt.plot(points_2[:,0], points_2[:,1], marker='o', color='red', markersize=4, ls='None', label = 'Cat')
        plt.legend()
       
        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.title('Raw')
        
        ax2 = figs.add_subplot(1,2,2) # nrows, ncols, plotnum. Returns an 'axis'
        plt.plot(points_1[:,0], points_1[:,1], marker='o', color='lightgreen', markersize=9, ls='None')
        plt.plot(points_2[:,0] + t0[1], points_2[:,1] + t0[0], marker='o', color='red', markersize=4, ls='None')
        ax2.set_aspect('equal')

        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.title('Shifted, dx=' + repr(dx) + ', dy = ' + repr(dy))
        
        plt.show()
        
    return t0