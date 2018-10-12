def fix_reqid(reqid):
    """
    Standardize the ReqID field so that it can be easily sorted alphabetically.
    Changes from JELR_02_RPHASE05 → JELR_RPHASE05_02
    """
    sub = reqid.split('_')
    if len(sub) == 2:        
        (p1,p2,p3) = (sub[0], '00', sub[1])
        
    else:        
        (p1,p2,p3) = reqid.split('_')
        
    out = f'{p1}_{p3}_{p2}'
        
    return out    
 
def get_fits_info_from_files_lorri(path,
                            file_tm = "/Users/throop/gv/dev/gv_kernels_new_horizons.txt", pattern=''):
    "Populate an astropy table with info from the headers of a list of LORRI files."
    import numpy as np
    import spiceypy as sp
    import glob
    import astropy
    from astropy.io import fits
    from astropy.table import Table
    import astropy.table
    import math
    import hbt
    

# For testing:
# file = '/Users/throop/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0035020322_0x630_sci_1.fit' # 119 deg phase as per gv
# file = '/Users/throop/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034599122_0x630_sci_1.fit' # 7 deg phase, inbound

# t = hbt.get_fits_info_from_files_lorri(file)

# Flags: Do we do all of the files? Or just a truncated subset of them, for testing purposes?
    
    DO_TRUNCATED = False
    NUM_TRUNC = 100

# We should work to standardize this, perhaps allowing different versions of this function 
# for different instruments.

    d2r = np.pi /180.
    r2d = 1. / d2r

    sp.furnsh(file_tm)

# *** If path ends with .fit or .fits, then it is a file not a path. Don't expand it, but read it as a single file.

    if (('.fits' in path) or ('.fit' in path)):
        file_list = path
        files = [file_list]

    else:
        
        dir_data = path          
    #dir_data = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all'
    # Start up SPICE
    
    
    # Get the full list of files
    # List only the files that match an (optional) user-supplied pattern, such as '_opnav'
    
        file_list = glob.glob(dir_data + '/*' + pattern + '.fit')
        files = np.array(file_list)
        indices = np.argsort(file_list)
        files = files[indices]

# Read the JD from each file. Then sort the files based on JD.

    jd = []
    for file in files:
        hdulist = fits.open(file)
        jd.append(hdulist[0].header['MET'])
        hdulist.close()
         
    fits_met     = [] # new list (same as array) 
    fits_startmet= [] 
    fits_stopmet = []
    fits_exptime = [] # starting time of exposure
    fits_target  = [] 
    fits_reqdesc = []     
    fits_reqcomm = [] # New 9-Oct-2018
    fits_reqid   = [] # New 9-Oct-2018
    fits_spcinst0= [] 
    fits_spcutcjd= []   
    fits_naxis1=   [] 
    fits_naxis2 =  []
    fits_sformat = [] # Data format -- '1x1' or '4x4'
    fits_spctscx = [] # sc - target, dx 
    fits_spctscy = [] # dy
    fits_spctscz = [] # dz
    fits_spctcb  = [] # target name
    fits_spctnaz = [] # Pole angle between target and instrument (i.e., boresight rotation angle)
    fits_rsolar  = [] # (DN/s)/(erg/cm^2/s/Ang/sr), Solar spectrum. Use for resolved sources.
    
    if (DO_TRUNCATED):
        files = files[0:NUM_TRUNC]
        
#files_short = np.array(files)
#for i in range(files.size):
#    files_short = files[i].split('/')[-1]  # Get just the filename itself

# Set up one iteration variable so we don't need to create it over and over
    num_obs = np.size(files)
    i_obs = np.arange(num_obs)
    
    print("Read " + repr(np.size(files)) + " files.")
    
    for file in files:
        print("Reading file " + file)
    
        hdulist = fits.open(file)
        header = hdulist[0].header
        
        keys = header.keys()
    
        fits_met.append(header['MET'])
        fits_exptime.append(header['EXPTIME'])
        fits_startmet.append(header['STARTMET'])
        fits_stopmet.append(header['STOPMET'])
        fits_target.append(header['TARGET'])
        fits_reqdesc.append(header['REQDESC'])
        fits_reqcomm.append(header['REQCOMM'])
        fits_reqid.append(header['REQID'])
        fits_spcinst0.append(header['SPCINST0'])
        fits_spcutcjd.append( (header['SPCUTCJD'])[3:]) # Remove the 'JD ' from before number
        fits_naxis1.append(header['NAXIS1'])
        fits_naxis2.append(header['NAXIS2'])
        fits_spctscx.append(header['SPCTSCX'])
        fits_spctscy.append(header['SPCTSCY'])
        fits_spctscz.append(header['SPCTSCZ'])    
        fits_spctnaz.append(header['SPCTNAZ'])    
        fits_sformat.append(header['SFORMAT'])
        fits_rsolar.append(header['RSOLAR'])   # NB: This will be in the level-2 FITS, but not level 1
                                             
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
    reqid      = np.array(fits_reqid)
    reqcomm    = np.array(fits_reqcomm)
    met0       = np.array(fits_startmet)
    met1       = np.array(fits_stopmet)
    exptime    = np.array(fits_exptime)
    rotation   = np.array(fits_spctnaz)
    sformat    = np.array(fits_sformat)
    rotation   = np.rint(rotation).astype(int)  # Turn rotation into integer. I only want this to be 0, 90, 180, 270... 
    rsolar     = np.array(fits_rsolar)
    
    files_short = np.zeros(num_obs, dtype = 'U60')

# Now do some geometric calculations and create new values for a few fields

    dist_targ = np.sqrt(dx_targ**2 + dy_targ**2 + dz_targ**2)

    phase = np.zeros(num_obs)
    utc = np.zeros(num_obs, dtype = 'U30')
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
# *** No, don't do that. The actual MET field is used for timestamping -- keep it as integer.

#    met = (met0 + met1) / 2.

# Loop over all images

    for i in i_obs:
    
# Get the ET and UTC, from the JD. These are all times *on s/c*, which is what we want
    
      et[i] = sp.utc2et('JD ' + repr(jd[i]))
      utc[i] = sp.et2utc(et[i], 'C', 2)
    
# Calculate Sun-Jupiter-NH phase angle for each image 
    
      (st_jup_sc, ltime) = sp.spkezr('Jupiter', et[i], frame, abcorr, 'New Horizons') #obs, targ
      (st_sun_jup, ltime) = sp.spkezr('Sun', et[i], frame, abcorr, 'Jupiter')
      ang_scat = sp.vsep(st_sun_jup[0:3], st_jup_sc[0:3])
      phase[i] = math.pi - ang_scat
#      phase[i] = ang_scat
      files_short[i] = files[i].split('/')[-1]
# Calc sub-sc lon/lat
      
      mx = sp.pxform(frame,'IAU_JUPITER', et[i])
      st_jup_sc_iau_jup = sp.mxv(mx, st_jup_sc[0:3])
      
      (radius,subsclon[i],subsclat[i]) = sp.reclat(st_jup_sc[0:3])  # Radians
      (radius,subsclon[i],subsclat[i]) = sp.reclat(st_jup_sc_iau_jup)  # Radians

# Stuff all of these into a Table

    t = Table([i_obs, met, utc, et, jd, files, files_short, naxis1, naxis2, target, instrument, 
               dx_targ, dy_targ, dz_targ, reqid, 
               met0, met1, exptime, phase, subsclat, subsclon, naxis1, 
               naxis2, rotation, sformat, rsolar, desc, reqcomm], 
               
               names = ('#', 'MET', 'UTC', 'ET', 'JD', 'Filename', 'Shortname', 'N1', 'N2', 'Target', 'Inst', 
                        'dx', 'dy', 'dz', 'ReqID',
                        'MET Start', 'MET End', 'Exptime', 'Phase', 'Sub-SC Lat', 'Sub-SC Lon', 'dx_pix', 
                        'dy_pix', 'Rotation', 'Format', 'RSolar', 'Desc', 'Comment'))
    
# Define units for a few of the columns
                        
    t['Exptime'].unit = 's'
    t['Sub-SC Lat'].unit = 'degrees'

# Create a dxyz_targ column, from dx dy dz. Easy!

    t['dxyz'] = np.sqrt(t['dx']**2 + t['dy']**2 + t['dz']**2)  # Distance, in km

    return t


if (__name__ == '__main__'):
    dir_images = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all'
    t = hbt.get_fits_info_from_files_lorri(dir_images, pattern='opnav')
    print(t)

    # Now calculate the group names, using my classic scheme my sorting on the Description field.
    
    groups_all = astropy.table.unique(t, keys=(['Desc']))['Desc']
    
    # Create new columns in the table: groupnum, imagenum
    
    t['groupnum'] = np.zeros(len(t)).astype(int)
    t['imagenum'] = np.zeros(len(t)).astype(int)

    # Standardize the reqid field, so it is sortable
    
    t['ReqID_fixed'] = t['ReqID']
    
    for i in range(len(t)):
        t['ReqID_fixed'][i] = fix_reqid(t['ReqID'][i])
        
    # Loop over all the groups, and assign a group and image number to each file
    
    for groupnum,group in enumerate(groups_all):
        is_match = t['Desc'] == group
        t['groupnum'][is_match] = groupnum
        t['imagenum'][is_match] = hbt.frange(0,np.sum(is_match)-1).astype(int)

    # Now get a list of all reqid's
    
    reqids_all = astropy.table.unique(t, keys=(['ReqID_fixed']))['ReqID']
    
    # Now get a list of reqid's, for rings only!
    
    groups_rings = groups_all[5:]
    groupmask = np.logical_or( (t['Desc'] == groups_rings[0]),
                               (t['Desc'] == groups_rings[1]))
    groupmask = np.logical_or(groupmask, t['Desc'] == groups_rings[2])
    groupmask = np.logical_or(groupmask, t['Desc'] == groups_rings[3])
    groupmask_rings = groupmask
    
    t_rings = t[groupmask_rings]
    reqids_rings = astropy.table.unique(t_rings, keys=(['ReqID_fixed']))['ReqID_fixed']    

    # Loop over all reqid's. For each one, print a line with some info

    reqids = reqids_rings
 
    # t_rings['RedID_fix'] = reqids_fix
    
    # reqids_sort = sorted(reqids_sort)

# # Stuff all of these into a Table

#     t = Table([i_obs, met, utc, et, jd, files, files_short, naxis1, naxis2, target, instrument, 
#                dx_targ, dy_targ, dz_targ, reqid, 
#                met0, met1, exptime, phase, subsclat, subsclon, naxis1, 
#                naxis2, rotation, sformat, rsolar, desc, reqcomm], 
               
#                names = ('#', 'MET', 'UTC', 'ET', 'JD', 'Filename', 'Shortname', 'N1', 'N2', 'Target', 'Inst', 
#                         'dx', 'dy', 'dz', 'ReqID',
#                         'MET Start', 'MET End', 'Exptime', 'Phase', 'Sub-SC Lat', 'Sub-SC Lon', 'dx_pix', 
#                         'dy_pix', 'Rotation', 'Format', 'RSolar', 'Desc', 'Comment'))
    
    for i,reqid in enumerate(reqids):

        is_match = t['ReqID_fixed'] == reqid
        num_images = np.sum(is_match)
        index_first = np.where(is_match)[0][0]
        index_last  = np.where(is_match)[0][-1]
        
        str_imagenum = f"{t['groupnum'][index_first]}/{t['imagenum'][index_first]}-{t['imagenum'][index_last]}"
        # str_last  = f"{t['groupnum'][index_last]}/{t['imagenum'][index_last]}"
        phase = t['Phase'][index_first] * hbt.r2d
        lat   = t['Sub-SC Lat'][index_first] * hbt.r2d
        et_start = t['ET'][index_first]
        et_end   = t['ET'][index_last]
        dt       = et_end - et_start
        ut       = t['UTC'][index_first][:-3]
        dist     = t['dxyz'][index_first] # Distance in km
        
        comment   = t['Comment'][index_first]
        desc      = t['Desc'][index_first]
        res       = 0.3*hbt.d2r/1024 * dist
        
        print(f'{reqid:18}; {num_images:3} ; {str_imagenum:9} ; {dist:6.0f}; {ut}; {dt:6.0f};' + \
              f' {phase:6.2f} ; {lat:6.2f} ; {desc} ; {comment}')

