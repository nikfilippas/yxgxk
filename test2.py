import numpy as np
import pymaster as nmt
from healpy.fitsfunc import read_map
from healpy.pixelfunc import ud_grade
from healpy.sphtfunc import anafast
from healpy.fitsfunc import read_alm
from healpy.sphtfunc import alm2map
import matplotlib.pyplot as plt


nside = 512

alm_f = "data/maps/COM_Lensing_4096_R3.00_MV_dat_klm.fits"
# map_f = "data/maps/COM_Lensing_4096_R3.00_MV_map.fits"
mask_f = "data/maps/COM_Lensing_4096_R3.00_mask.fits"

alms = read_alm(alm_f)
Map = alm2map(alms, nside)

# Map = read_map(map_f)
mask = read_map(mask_f)

Map, mask = map(lambda X: ud_grade(X, nside), [Map, mask])

field = nmt.NmtField(mask, [Map])
b = nmt.NmtBin(nside, nlb=10)

wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(field, field, b)

cl_coupled = nmt.compute_coupled_cell(field, field)
cl_decoupled = wsp.decouple_cell(cl_coupled)

# anafast
cl_ana = anafast(Map)
cl_ana /= mask.mean()

# Planck
planck_f  = "data/maps/COM_Lensing_4096_R3.00_MV_nlkk.dat"
planck = np.loadtxt(planck_f)


plt.loglog(b.get_effective_ells(), cl_decoupled.T)
plt.loglog(np.arange(cl_ana.size), cl_ana)
plt.loglog(planck[0:, 0], planck[0:, 2])
