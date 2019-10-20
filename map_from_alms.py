"""
Processes Planck alm tables and produces map.
"""

import sys
import numpy as np
from astropy.io import fits
from healpy.sphtfunc import alm2map
from healpy.fitsfunc import write_map

try:
    fname_dat = sys.argv[1]
    try:
        nside = sys.argv[2]
    except:
        print("`nside` not set. Defaulting to 2048.")
        nside = 2048
except:
    raise ValueError("Must provide data table name as command-line argument")


# fname_dat = "data/maps/COM_Lensing_4096_R3.00_MV_dat_klm.fits"

# columns: l*l+l+m+1 (ells); real; imag
fits_file = fits.open(fname_dat)[1]
header = fits_file.header
data = fits_file.data
# processing lists is ~ 5x faster
data = np.array(data.tolist())
alms = data[:, 1] + data[:, 2]*1j

map = alm2map(alms, nside=nside)

fname_map = fname_dat.split(".fits")[0]+"_map.fits"
write_map(fname_map, map, overwrite=True)