import numpy as np
import matplotlib.pyplot as plt

ls15, nl_P15, Cl_P15 = np.loadtxt("data/maps/COM_CompMap_Lensing_2048_R2.00_nlkk.dat").T
ls18, nl_P18, Cl_P18 = np.loadtxt("data/maps/COM_Lensing_4096_R3.00_MV_nlkk.dat").T
ls18sz, nl_P18sz, Cl_P18sz = np.loadtxt("data/maps/COM_Lensing_Szdeproj_4096_R3.00_TT_nlkk.dat").T

Cl_P15 -= nl_P15
Cl_P18 -= nl_P18
Cl_P18sz -= nl_P18sz

# # kappa x kappa model using CCL
# import pyccl as ccl
# from model.cosmo_utils import COSMO_DEFAULT
# cosmo = COSMO_DEFAULT()
# ell = np.geomspace(6, 3500, 200)
# cmbl = ccl.CMBLensingTracer(cosmo, z_source=1080)
# Cell = ccl.angular_cl(cosmo, cmbl, cmbl, ell)
# plt.figure()
# plt.loglog(ls18, Cl_P18, lw=1.5, label="Planck18")
# plt.loglog(ls18sz, Cl_P18sz, lw=1.5, label="Planck18 SZ")
# plt.loglog(ls15, Cl_P15, lw=1.5, label="Planck15")
# plt.loglog(ell, Cell, "k:", lw=1, label="CCL fit")
# plt.legend(loc="best")


# using HEALPix
import healpy as hp
hp.disable_warnings()
m15 = hp.fitsfunc.read_map("data/maps/COM_CompMap_Lensing_2048_R2.00_map.fits", dtype=float)
m18 = hp.fitsfunc.read_map("data/maps/COM_Lensing_4096_R3.00_MV_map.fits", dtype=float)
m18sz = hp.fitsfunc.read_map("data/maps/COM_Lensing_Szdeproj_4096_R3.00_TT_map.fits", dtype=float)

cl15hp = hp.anafast(m15, lmax=2048)
cl18hp = hp.anafast(m18, lmax=2048)
cl18szhp = hp.anafast(m18sz, lmax=2048)

plt.figure()
plt.loglog(np.arange(2049), cl15hp, "bo", ms=1, label="healpy Cl from map, P15")
plt.loglog(np.arange(2049), cl18hp, "yo", ms=1, label="healpy Cl from map, P18")
plt.loglog(np.arange(2049), cl18hp, "go", ms=1, label="healpy Cl from map, P18sz")
plt.loglog(ls18, Cl_P18+nl_P18, "y--", label="Planck data, P18")
plt.loglog(ls18sz, Cl_P18sz+nl_P18sz, "g--", label="Planck data, P18sz")
plt.loglog(ls15, Cl_P15+nl_P15, "b--", label="Planck data, P15")
plt.xlim(1, 2048)
plt.ylim(1e-7, 6e-6)
plt.legend(loc="best")
plt.savefig("anafast_all.pdf")


# using NaMaster
import pymaster as nmt
from analysis.bandpowers import Bandpowers
d = {"lsplit":52, "nb_log":28, "nlb":20, "nlb_lin":10, "type":"linlog"}
b = Bandpowers(2048, d).bn
wsp = nmt.NmtWorkspace()

msk = hp.fitsfunc.read_map("data/maps/COM_CompMap_Lensing_2048_R2.00_mask.fits", dtype=float)
field = nmt.NmtField(msk, [m15])
wsp.compute_coupling_matrix(field, field, b)
cl15nmt = wsp.decouple_cell(nmt.compute_coupled_cell(field, field)).T

msk = hp.fitsfunc.read_map("data/maps/COM_Lensing_4096_R3.00_mask.fits", dtype=float)
field = nmt.NmtField(msk, [m18])
wsp.compute_coupling_matrix(field, field, b)
cl18nmt = wsp.decouple_cell(nmt.compute_coupled_cell(field, field)).T

msk = hp.fitsfunc.read_map("data/maps/COM_Lensing_Szdeproj_4096_R3.00_mask.fits", dtype=float)
field = nmt.NmtField(msk, [m18])
wsp.compute_coupling_matrix(field, field, b)
cl18sznmt = wsp.decouple_cell(nmt.compute_coupled_cell(field, field)).T

plt.figure()
plt.loglog(b.get_effective_ells(), cl15nmt, "bo", ms=1, label="NaMaster Cl from map, P15")
plt.loglog(b.get_effective_ells(), cl18nmt, "yo", ms=1, label="NaMaster Cl from map, P18")
plt.loglog(b.get_effective_ells(), cl18sznmt, "go", ms=1, label="NaMaster Cl from map, P18sz")
plt.loglog(ls18, Cl_P18+nl_P18, "y--", label="Planck data, P18")
plt.loglog(ls18sz, Cl_P18sz+nl_P18sz, "g--", label="Planck data, P18sz")
plt.loglog(ls15, Cl_P15+nl_P15, "b--", label="Planck data, P15")
plt.xlim(1, 2048)
plt.ylim(1e-7, 6e-6)
plt.legend(loc="best")
plt.savefig("namaster_all.pdf")
