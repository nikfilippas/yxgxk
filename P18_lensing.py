import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from analysis.params import ParamRun

# SZ-deprojected
d = np.loadtxt("data/maps/COM_Lensing_Szdeproj_4096_R3.00_TT_nlkk.dat")
ls, nl_P18, Cl_P18 = d.T
Cl_P18 -= nl_P18

# CCL
fname_params = "params_lensing.yml"
p = ParamRun(fname_params)
cosmo = p.get_cosmo()
cmbl = ccl.CMBLensingTracer(cosmo, z_source=1100)
ell = np.geomspace(6, 3500, 200)
Cell = ccl.angular_cl(cosmo, cmbl, cmbl, ell)

plt.figure()
plt.loglog(ls, Cl_P18, lw=3, label="Planck18")
plt.loglog(ell, Cell, "k--", lw=2, label="CCL fit")
plt.legend(loc="best")

# # pipeline
# from model.profile2D import Lensing
# from model.power_spectrum import hm_ang_power_spectrum
# from model.hmcorr import HaloModCorrection
# k = Lensing()
# cosmo_pars = p.get_cosmo_pars()
# Cl = hm_ang_power_spectrum(ell, (k,k),
#                            zpoints=64,
#                            hm_correction=HaloModCorrection,
#                            **{**cosmo_pars, "a_HMcorr": 0.35015861})

# plt.loglog(ell, Cl, "r:", label="pipeline")
# plt.legend(loc="best")

# measurement from map
measurement = np.load("measurement.npz")
leff, cell = measurement["arr_0"], measurement["arr_1"]
plt.loglog(leff, cell, "ro", ms=2, label="nmt from map")
plt.loglog(ls, Cl_P18+nl_P18, "r--", label="Planck w/ noise")
plt.legend(loc="best")