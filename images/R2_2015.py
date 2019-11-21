"""
plot measured data from NaMaster via pipeline.py versus Planck data
Tests direct map output via NaMaster for cls versus Planck nlkk for R2 2015.
"""
import numpy as np
import pymaster as nmt
from healpy.fitsfunc import read_map
from healpy.pixelfunc import ud_grade
import matplotlib.pyplot as plt


nside = 2048

## alms ##
# from healpy.fitsfunc import read_alm
# from healpy.sphtfunc import alm2map
# alm_f = "data/maps/COM_CompMap_Lensing_2048_R2.00_dat_klm.fits"
# alms = read_alm(alm_f)
# Map = alm2map(alms, nside)

## map ##
map_f = "../data/maps/COM_CompMap_Lensing_2048_R2.00_map.fits"
Map = read_map(map_f)

## mask ##
mask_f = "../data/maps/COM_Lensing_4096_R3.00_mask.fits"
mask = read_map(mask_f)

Map, mask = map(lambda X: ud_grade(X, nside), [Map, mask])

field = nmt.NmtField(mask, [Map])
b = nmt.NmtBin(nside, nlb=10)

wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(field, field, b)

cl_coupled = nmt.compute_coupled_cell(field, field)
cl_decoupled = wsp.decouple_cell(cl_coupled)


# Planck
planck_f  = "../data/maps/COM_CompMap_Lensing_2048_R2.00_nlkk.dat"
planck = np.loadtxt(planck_f)
plt.loglog(b.get_effective_ells(), cl_decoupled.T, label="nmt")
plt.loglog(planck[:, 0], planck[:, 2], label="planck")
plt.fill_between(planck[:, 0], planck[:, 2]-planck[:, 1],
                 planck[:, 2]+planck[:, 1], color="orange", alpha=0.2)
plt.ylim(cl_decoupled.T.min(),)
plt.xlabel(r"$\ell$")
plt.legend(loc="best")
plt.savefig("nmt_R2_2015_comparison.pdf")


# # anafast
# from healpy.sphtfunc import anafast
# cl_ana = anafast(Map)
# cl_ana /= mask.mean()
# plt.loglog(np.arange(cl_ana.size), cl_ana)
