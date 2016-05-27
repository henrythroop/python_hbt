def get_fits_info_from_files_lorri(path,
                            file_tm = "/Users/throop/gv/dev/gv_kernels_new_horizons.txt" ):
    "Populate an astropy table with info from the headers of a list of LORRI files."
    import numpy as np
    import cspice
    import glob
    import astropy
    from astropy.io import fits
    from astropy.table import Table
    import astropy.table

# Flags: Do we do all of the files? Or just a truncated subset of them, for testing purposes?
    
    DO_TRUNCATED = True
    NUM_TRUNC = 100


# We should work to standardize this, perhaps allowing different versions of this function 
# for different instruments.

    d2r = np.pi /180.
    r2d = 1. / d2r

    dir_data = path          
#dir_data = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all'
# Start up SPICE

    cspice.furnsh(file_tm)

# Get the full list of files

    file_list = glob.glob(dir_data + '/*fit')
    files = np.array(file_list)
    indices = np.argsort(file_list)
    files = files[indices]

# Read the JD from each file. Then sort the files based on JD.

    jd = []
    for file in files:
        hdulist = fits.open(file)
        jd.append(hdulist[0].header['MET'])
        hdulist.close()
         
    fits_met = []    # new list (same as array) 
    fits_startmet  = [] 
    fits_stopmet= []
    fits_exptime = [] # starting time of exposure
    fits_target  = [] 
    fits_reqdesc = [] 
    fits_spcinst0= [] 
    fits_spcutcjd= []   
    fits_naxis1= [] 
    fits_naxis2 = []
    fits_sformat = [] # Data format -- '1x1' or '4x4'
    fits_spctscx = [] # sc - target, dx 
    fits_spctscy = [] # dy
    fits_spctscz = [] # dz
    fits_spctcb  = [] # target name
    fits_spctnaz = [] # Pole angle between target and instrument (i.e., boresight rotation angle)

    if (DO_TRUNCATED):
        files = files[0:NUM_TRUNC]
        
#files_short = np.array(files)
#for i in range(files.size):
#    files_short = files[i].split('/')[-1]  # Get just the filename itself

# Set up one iteration variable so we don't need to create it over and over
    num_obs = np.size(files)
    i_obs = np.arange(num_obs)
    
    for file in files:
        print "Reading file " + file
    
        hdulist = fits.open(file)
        header = hdulist[0].header
        
        keys = header.keys()
    
        fits_met.append(header['MET'])
        fits_exptime.append(header['EXPTIME'])
        fits_startmet.append(header['STARTMET'])
        fits_stopmet.append(header['STOPMET'])
        fits_target.append(header['TARGET'])
        fits_reqdesc.append(header['REQDESC'])
        fits_spcinst0.append(header['SPCINST0'])
        fits_spcutcjd.append( (header['SPCUTCJD'])[3:]) # Remove the 'JD ' from before number
        fits_naxis1.append(header['NAXIS1'])
        fits_naxis2.append(header['NAXIS2'])
        fits_spctscx.append(header['SPCTSCX'])
        fits_spctscy.append(header['SPCTSCY'])
        fits_spctscz.append(header['SPCTSCZ'])    
        fits_spctnaz.append(header['SPCTNAZ'])    
        fits_sformat.append(header['SFORMAT'])    
           
        hdulist.close() # Close the FITS file

#print object
#print "done"

# Calculate distance to Jupiter in each of these
# Calc phase angle (to Jupiter)
# Eventually build backplanes: phase, RA/Dec, etc.
# Eventually Superimpose a ring on top of these
#  ** Not too hard. I already have a routine to create RA/Dec of ring borders.
# Eventually overlay stars 
#   Q: Will there be enough there?
# Eventually repoint based on stars
#  ** Before I allow repointing, I should search a star catalog and plot them.

# Convert some things to numpy arrays. Is there any disadvantage to this?

    met        = np.array(fits_met)
    jd         = np.array(fits_spcutcjd, dtype='d') # 'f' was rounding to one decimal place...
    naxis1     = np.array(fits_naxis1)
    naxis2     = np.array(fits_naxis2)
    target     = np.array(fits_target) # np.array can use string arrays as easily as float arrays
    instrument = np.array(fits_spcinst0)
    dx_targ    = np.array(fits_spctscx)
    dy_targ    = np.array(fits_spctscy)
    dz_targ    = np.array(fits_spctscz)
    desc       = np.array(fits_reqdesc)
    met0       = np.array(fits_startmet)
    met1       = np.array(fits_stopmet)
    exptime    = np.array(fits_exptime)
    rotation   = np.array(fits_spctnaz)
    sformat    = np.array(fits_sformat)
    rotation   = np.rint(rotation).astype(int)  # Turn rotation into integer. I only want this to be 0, 90, 180, 270... 
    files_short = np.zeros(num_obs, dtype = 'S30')

# Now do some geometric calculations and create new values for a few fields

    dist_targ = np.sqrt(dx_targ**2 + dy_targ**2 + dz_targ**2)

    phase = np.zeros(num_obs)
    utc = np.zeros(num_obs, dtype = 'S30')
    et = np.zeros(num_obs)
    subsclat = np.zeros(num_obs) # Sub-sc latitude
    subsclon = np.zeros(num_obs) # Sub-sc longitude
    
    name_observer = 'New Horizons'
    frame = 'J2000'
    abcorr = 'LT+S'
#         Note that using light time corrections alone ("LT") is 
#         generally not a good way to obtain an approximation to an 
#         apparent target vector:  since light time and stellar 
#         aberration corrections often partially cancel each other, 
#         it may be more accurate to use no correction at all than to 
#         use light time alone. 

# Fix the MET. The 'MET' field in fits header is actually not the midtime, but the time of the first packet.
# I am going to replace it with the midtime.

    met = (met0 + met1) / 2.

# Loop over all images

    for i in i_obs:
    
# Get the ET and UTC, from the JD. These are all times *on s/c*, which is what we want
    
      et[i] = cspice.utc2et('JD ' + repr(jd[i]))
      utc[i] = cspice.et2utc(et[i], 'C', 2)
    
# Calculate Sun-Jupiter-NH phase angle for each image 
    
      (st_jup_sc, ltime) = cspice.spkezr('Jupiter', et[i], frame, abcorr, 'New Horizons') #obs, targ
      (st_sun_jup, ltime) = cspice.spkezr('Sun', et[i], frame, abcorr, 'Jupiter')
      phase[i] = cspice.vsep(st_sun_jup[0:3], st_jup_sc[0:3])
      files_short[i] = files[i].split('/')[-1]
# Calc sub-sc lon/lat
      
      (radius,subsclon[i],subsclat[i]) = cspice.reclat(st_jup_sc[0:3])

# Stuff all of these into a Table

    t = Table([i_obs, met, utc, et, jd, files, files_short, naxis1, naxis2, target, instrument, 
               dx_targ, dy_targ, dz_targ, desc, 
               met0, met1, exptime, phase, subsclat, subsclon, naxis1, 
               naxis2, rotation, sformat], 
               
               names = ('#', 'MET', 'UTC', 'ET', 'JD', 'Filename', 'Shortname', 'N1', 'N2', 'Target', 'Inst', 
                        'dx', 'dy', 'dz', 'Desc',
                        'MET Start', 'MET End', 'Exptime', 'Phase', 'Sub-SC Lat', 'Sub-SC Lon', 'dx_pix', 
                        'dy_pix', 'Rotation', 'Format'))
    
# Define units for a few of the columns
                        
    t['Exptime'].unit = 's'
    t['Sub-SC Lat'].unit = 'degrees'

# Create a dxyz_targ column, from dx dy dz. Easy!

    t['dxyz'] = np.sqrt(t['dx']**2 + t['dy']**2 + t['dz']**2)

    return t
