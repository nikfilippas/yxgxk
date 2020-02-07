"""compares yxg to yxgxk"""

## GLOBAL PARAMETERS ##
import numpy as np
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665,
                      mass_function="tinker")

kwargs = {'fc': 1.0,
          'alpha': 1.0,
          'beta_gal': 1.0,
          'sigma_lnM': 0.15,
          'M1': 12.939399020269125,
          'Mmin': 11.616598183948195,
          'b_hydro': 0.7878221462263877,
          'r_corr': 0.6762753732566117,
          'width': 1.1981397977341839,
          'M0': 11.616598183948195,
          'beta_max': 1.0,
          'Omega_c': 0.26066676,
          'Omega_b': 0.048974682,
          'h': 0.6766,
          'sigma8': 0.8102,
          'n_s': 0.9665,
          'mass_function': 'tinker',
          'a_HMcorr': 0.35015861}


## HM CORRECTION
# old
from model.hmcorr import HalomodCorrection_old
hm_correction = HalomodCorrection_old(cosmo).rk_interp
# new
from model.hmcorr import HaloModCorrection


## PREVIOUS ANALYSIS ##
# gxg data
lsd0 = np.load("legacy/lsd0.npy")
dd0 = np.load("legacy/dd0.npy")
# yxg data
lsd1 = np.load("legacy/lsd1.npy")
dd1 = np.load("legacy/dd1.npy")

# gxg theory
lst0 = np.load("legacy/lst0.npy")
tt0 = np.load("legacy/tt0.npy")
# yxg theory
lst1 = np.load("legacy/lst1.npy")
tt1 = np.load("legacy/tt1.npy")


## NEW ANALYSIS ##
# gxg data
LSD0 = np.load("output_default/cls_2mpz_2mpz.npz")
# yxg data
LSD1 = np.load("output_default/cls_2mpz_y_milca.npz")

# global theory
L = np.arange(6, 3000, 1)
from model.power_spectrum import hm_ang_power_spectrum
from model.profile2D import HOD, Arnaud
g = HOD(nz_file="data/dndz/2MPZ_bin1.txt")
y = Arnaud()
# gxg theory
TT0 = hm_ang_power_spectrum(L, (g, g), hm_correction=hm_correction, **kwargs)
TT0_HM = hm_ang_power_spectrum(L, (g, g), hm_correction=HaloModCorrection, **kwargs)
# yxg theory
TT1 = hm_ang_power_spectrum(L, (y, g), hm_correction=HaloModCorrection, **kwargs)


## FIGURE ###
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)

# gxg #
# old data
ax[0].plot(lsd0, dd0, "ro", alpha=0.2)
# old theory
ax[0].plot(lst0, tt0, "k--", alpha=0.3, lw=3, label="yxg fit")
# new data
ax[0].plot(LSD0["ls"], LSD0["cls"], "bo", alpha=0.2)
# new theory
ax[0].plot(L, TT0, "k:", alpha=0.3, lw=3, label="new fit, old HMcorr")
ax[0].plot(L, TT0_HM, "k-", alpha=0.3, lw=3, label="new fit")
ax[0].legend(loc="upper right")

# yxg #
# old data
ax[1].plot(lsd1, dd1, "ro", alpha=0.2)
# old theory
ax[1].plot(lst1, tt1, "k--", alpha=0.3, lw=3, label="yxg fit")
# new data
ax[1].plot(LSD1["ls"], LSD1["cls"], "bo", alpha=0.2)
# new theory
ax[1].plot(L, TT1, "k-", alpha=0.3, lw=3, label="new fit")

for a in ax:
    a.loglog()
    a.legend(loc="upper right")

# fig.savefig("comparison.pdf")
