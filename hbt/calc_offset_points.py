#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:52:17 2016

@author: throop
"""

##########
# Calc offset between two sets of points. Generates an image for each one, and then calcs relative image shift.
##########

def calc_offset_points(points_1, points_2, shape, diam_kernel = 5, do_plot=False):
    import hbt
    import matplotlib.pyplot as plt
    import imreg_dft as ird

    "Calculate the offset between a pair of ordered points -- e.g., an xy list"
    "of star positions, and and xy list of model postns."
    "Returned offset is integer pixels as tuple (dy, dx)."
    
#    diam_kernel = 5 # Set the value of the fake stellar image to plot
                    # diam_kernel = 5 is best for LORRI. 11 is too big, and we get the wrong answer. Very sensitive.

    image_1 = hbt.image_from_list_points(points_1, shape, diam_kernel)
    image_2 = hbt.image_from_list_points(points_2, shape, diam_kernel)
 
    (dy,dx) = ird.translation(image_1, image_2)['tvec'] # Return shift, with t0 = (dy, dx). 
                                                        # ** API changed ~ Sep-16, Anaconda 4.2?
 
    plt.imshow(image_1)
    plt.title('Image 1')
    plt.show()
    
    plt.imshow(image_2)
    plt.title('Image 2')
    plt.show()
    
    print("dx={}, dy={}".format(dx,dy))
    
    if (do_plot):

        print("DO_PLOT set") 

        xrange = (0, shape[0]) # Set xlim (aka xrange) s.t. 
        yrange = (shape[1], 0)

        # Do some plots. But for reasons I don't understand, only the last of these plots is actually going to screen?
        
#        figs = plt.figure()
#        ax1 = figs.add_subplot(1,2,1) # nrows, ncols, plotnum. Returns an 'axis'
#        plt.set_aspect('equal') # Need to explicitly set aspect ratio here, otherwise in a multi-plot, it will be rectangular

        plt.plot(points_1[:,0], points_1[:,1], marker='o', color='pink', markersize=4, ls='None', label = 'Photometric')
        plt.plot(points_2[:,0], points_2[:,1], marker='o', color='lightrgreen', markersize=4, ls='None', label = 'Cat')
        plt.legend(framealpha=0.5)
       
        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.title('Raw')
        plt.show()

        
         # Do this plot again a second time.
         
        plt.plot(points_1[:,0], points_1[:,1], marker='o', color='pink', markersize=4, ls='None', label = 'Photometric')
        plt.plot(points_2[:,0], points_2[:,1], marker='o', color='lightrgreen', markersize=4, ls='None', label = 'Cat')
        plt.legend(framealpha=0.5)
       
        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.title('Raw')
        plt.show()

        
        plt.plot(points_1[:,0], points_1[:,1], marker='o', color='lightgreen', markersize=9, ls='None')
        plt.plot(points_2[:,0] + dx, points_2[:,1] + dy, marker='o', color='red', markersize=4, ls='None')
#        plt.aspset_aspect('equal')

        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.title('Shifted, dx=' + repr(dx) + ', dy = ' + repr(dy))
        
        plt.show()

        
    return (dy, dx)