"""compares yxg to yxgxk"""

import numpy as np
import matplotlib.pyplot as plt

# gxg
lsd0 = np.load("legacy/lsd0.npy")
dd0 = np.load("legacy/dd0.npy")

lst0 = np.load("legacy/lst0.npy")
tt0 = np.load("legacy/tt0.npy")


plt.figure(1)
plt.loglog()
plt.plot(lsd0, dd0, "ro", alpha=0.2)
plt.plot(lst0, tt0, "k--", alpha=0.3, lw=3, label="yxg fit")


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


# yxgxk
LSD0 = np.load("output_default/cls_2mpz_2mpz.npz")
plt.plot(LSD0["ls"], LSD0["cls"], "bo", alpha=0.2)

from model.profile2D import HOD
g = HOD(nz_file="data/dndz/2MPZ_bin1.txt")

from model.power_spectrum import hm_ang_power_spectrum
from model.hmcorr import HalomodCorrection_old
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665,
                      mass_function="tinker")
hm_correction = HalomodCorrection_old(cosmo).rk_interp

L = np.arange(6, 3000, 1)
TT0 = hm_ang_power_spectrum(L, (g, g),
                            hm_correction=hm_correction,
                            **kwargs)

plt.plot(L, TT0, "k:", alpha=0.3, lw=3, label="new fit, old HMcorr")


# new HM correction
from model.hmcorr import HaloModCorrection
TT0_HM = hm_ang_power_spectrum(L, (g, g),
                            hm_correction=HaloModCorrection,
                            **kwargs)

plt.plot(L, TT0_HM, "k-", alpha=0.3, lw=3, label="new fit")
plt.legend(loc="best")
plt.savefig("comparison.pdf")
