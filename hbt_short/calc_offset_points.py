#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:52:17 2016

@author: throop
"""

##########
# Calc offset between two sets of points. Generates an image for each one, and then calcs relative image shift.
##########

def calc_offset_points(points_1, points_2, shape, diam_kernel = 9, labels=['', ''], 
                                do_binary = True, do_plot_before=False, do_plot_after=False, do_plot_raw=False):
    """
    points_1, points_2: 
    """
    
    import hbt
    import matplotlib.pyplot as plt
    import imreg_dft as ird
    import numpy as np

    """ 
    Calculate the offset between a pair of ordered points -- e.g., an xy list
    of star positions, and and xy list of model postns.
    Returned offset is integer pixels as tuple (dy, dx).
    Input lists are of shape N x 2.
    Y = column 0
    X = column 1
    The sizes of the two lists do not need to be identical.
    """
    
#    diam_kernel = 5 # Set the value of the fake stellar image to plot
                    # diam_kernel = 5 is best for LORRI. 11 is too big, and we get the wrong answer. Very sensitive.

    diam_kernel = 9
    
    image_1 = hbt.image_from_list_points(points_1, shape, diam_kernel, do_binary=do_binary)
    image_2 = hbt.image_from_list_points(points_2, shape, diam_kernel, do_binary=do_binary)
 
#    (dy,dx) = get_image_translation(image_1, image_2)
    
    # Get the shift, using FFT method
    
    (dy,dx) = ird.translation(image_1, image_2)['tvec'] # Return shift, with t0 = (dy, dx). 
                                                        # ** API changed ~ Sep-16, Anaconda 4.2?

#    DO_PLOT_INPUT_FRAMES = False  
    
    if (do_plot_raw): # Plot the raw frames generated to calculate the shift
        plt.imshow(image_1)
        plt.title('Image 1 = ' + labels[0] + ', diam_kernel = {}'.format(diam_kernel))
        plt.show()
        
        plt.imshow(image_2)
        plt.title('Image 2 = ' + labels[1])
        plt.show()
        
        plt.imshow(image_1 + image_2)
        plt.title('Image 1+2 = ' + labels[1])
        plt.show()        
    
    print("dx={}, dy={}".format(dx,dy))
    
    if (do_plot_before):

        xrange = (0, shape[0]) # Set xlim (aka xrange) s.t. 
        yrange = (shape[1], 0)
#        yrange = (0, shape[1])

        plt.plot(points_1[:,1], points_1[:,0], marker='o', color='none', markersize=10, ls='None', 
                 label = labels[0], mew=1, mec='red')
        plt.plot(points_2[:,1], points_2[:,0], marker='o', color='lightgreen', markersize=4, ls='None', 
                 label = labels[1])
        plt.title('Before shift of dx={:.1f}, dy={:.1f}'.format(dx, dy))
        plt.legend(framealpha=0.5)
#        plt.set_aspect('equal')
       
        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.show()

    if (do_plot_after):

        xrange = (0, shape[0]) # Set xlim (aka xrange) s.t. 
        yrange = (shape[1], 0)

        plt.plot(points_1[:,1], points_1[:,0], marker='o', color='none', markersize=10, ls='None', 
                 label = labels[0], mec='red', mew=1)
        plt.plot(points_2[:,1] + dy, points_2[:,0] + dx, marker='o', color='lightgreen', markersize=4, ls='None', 
                 label = labels[1])
        plt.legend(framealpha=0.5)
        plt.title('After shift of dx={:.1f}, dy={:.1f}'.format(dx, dy))
       
        plt.xlim(xrange)    # Need to set this explicitly so that points out of image range are clipped
        plt.ylim(yrange)
        plt.show()
        
    return (dy, dx)