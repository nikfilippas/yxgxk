"""
Processes Planck alm tables and produces maps.
"""

import os
from healpy.fitsfunc import read_alm
from healpy.sphtfunc import alm2map
from healpy.fitsfunc import write_map

d0 = "data/maps/"

files = os.popen("ls %s" % (d0+"*dat_klm.fits")).read().split("\n")[:-1]

nside = 2048

def make_map(fname_dat):
    """Makes map from alm file, optionally removes alm file."""
    # columns: l*l+l+m+1 (ells); real; imag
    alms = read_alm(fname_dat)
    Map = alm2map(alms, nside=nside)

    fname_map = fname_dat.split("dat_klm.fits")[0]+"map.fits"
    write_map(fname_map, Map, overwrite=True)
    print("    constructed map %s" % fname_map.split("/")[-1])
    # os.system("rm %s" % fname_dat)
    return None


for f in files:
    make_map(f)
