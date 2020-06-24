# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:37:07 2016

NH_JRING_CREATE_MEDIANS.PY

This routine creates median files, used for stray light subtraction.

@author: throop
"""

def nh_create_straylight_median_filename(index_group, index_files, do_fft=False, do_sfit=True, power=5):

    import numpy as np
    
    """
    Create the base filename for a stray light median file.
    Does not have an extension or a directory -- just the base name.
    
    index_group : Index of the group (scalar)
    index_files : Index of the files (array). Note that the filename is constructed from the start and end indices,
                  so it would possible to make two different arrays, with same filename, if arrays are not 
                  contiguous. To avoid this, make sure that arrays are inclusive of all elements.
    """
    
    # Usually we will add extensions onto this:
    #  .pkl -- save file
    #  .png -- file to be edited in photoshop
    
    str_type = ''

#==============================================================================
# String for the group number
#==============================================================================

    str_group = '_g{:.0f}'.format(index_group)

#==============================================================================
# String for the file number
#==============================================================================

# Determine if we are passed a scalar, or a vector (ie, individual file or list)
# Nice python trick: if we don't know if x is a scalar or vector or single-element vector, 
# just take np.amin(x). It will return a scalar, with no error.

    is_array =  (np.amin(index_files) != np.amax(index_files))

#    print("is_array = {}".format(is_array))
#    print("index_files = {}".format(index_files))
    
    if is_array: 
#        print ("is array!") 
        str_files = '_n{:.0f}..{:.0f}'.format(index_files[0], index_files[-1])
    else:
#        print("Is not array")
        str_files = '_n{:.0f}'.format(np.amin(index_files))

#==============================================================================
# String if FFT used
#==============================================================================
     
    if (do_fft):
        str_type += '_fft'

#==============================================================================
# String if power law used
#==============================================================================

    if (do_sfit):
        str_type += '_sfit{:.0f}'.format(power)

#==============================================================================
# Create the output string
#==============================================================================
        
    file_base = 'straylight_median' + str_group + str_files + str_type
    
    return file_base
    
    # NB: The print format string :.0f is useful for taking a float *or* int and formatting as an int.
    #     The 'd' string is for a decimal (int) but chokes on a float.
        
def test():
    import numpy as np

    s = nh_create_straylight_median_filename(7, np.array([4,100]), do_sfit = True, power = 3)
    s = nh_create_straylight_median_filename(1, 98, do_fft = True, power = 3)

    print(s)
    