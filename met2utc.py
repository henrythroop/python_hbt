def met2utc(met, name_observer):

# ; GV_MET2UTC.pro -- Convert from MET to UTC, for New Horizons only.
# ;
# ; Returns error message for any other mission.
# ;
# ; Assumes that the SPICE Leap seconds kernel and New Horizons
# ; spacecraft clock (SCLK) kernel have already been loaded. 
# 
# ; The NH sclk produces a "tick" every 20 microseconds, so there are
# ; 50,000 ticks in 1 second of MET
# ;
# ; Note, 1 "second" of MET is not quite equal to 1 second of time, since the
# ; spacecraft clock drifts a tiny amount. 
# ;
# ; A. Steffl, Mar 2008  -- Original version
# ; H. Throop, Apr 2008  -- Modified for GV
# ; H. Throop, Apr 2013  -- Changed sclk_ticks from double to float, as per Nathaniel Cunningham
# ; H. Throop, Jun 2015  -- Rewritten from IDL to python
# 
# ; common gv_common

# Relies on cspice already being up and running

    if ('NEW HORIZONS' not in name_observer):
	error = 'MET can be used only for New Horizons'
	print error
	return 0

    sclk_ticks = met * 5e4
    ntime = length(met)  # Vectorized
    et    = np.zeros(ntime) 
    utc   = np.zeros(ntime, dtype="S30")

    for i in range(ntime):
	et[i] = cspice.sct2e(-98, sclk_ticks[i])

	utc[i] = cspice.et2uct(et[i], 'ISOC', 3)

    if (ntime == 1): 
      utc = utc[0]

    return utc

# 
#   FOR i = 0, ntime-1 DO BEGIN  
#      CSPICE_SCT2E, -98, sclk_ticks[i], et_i
#      et[i] = et_i
#      IF NOT keyword_set(doy) THEN cspice_et2utc, et_i, 'ISOC', 3, utc_i ELSE $
#        cspice_et2utc, et_i, 'ISOD', 3, utc_i
#      utc[i] = utc_i
#   ENDFOR
#   
#   IF ntime EQ 1 THEN utc = utc[0]
#   return, utc
# 
# END
