"""
Find the best-fit values for the amplitude of the Gaussian
approximation to the halo model correction.
"""
from model.hmcorr import HM_Gauss
from model.cosmo_utils import COSMO_DEFAULT
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt

# define cosmology
cosmo = COSMO_DEFAULT()
kw = {"mass_function": ccl.halos.mass_function_from_name("tinker08"),
      "halo_bias": ccl.halos.halo_bias_from_name("tinker10")}

# load N(z)
z_arr = np.zeros(6)
zbins = ["2mpz"] + ["wisc%d" % i for i in range(1,6)]
for n, zbin in enumerate(zbins):
    z, nz = np.loadtxt("data/dndz/%s_DIR.txt" % zbin, unpack=True)
    z_arr[n] = np.average(z, weights=nz)
a_arr = 1/(1+z_arr)

# define halo model correction
hmcorr = HM_Gauss(cosmo, **kw)

# call amplitude best-fit function
print(np.column_stack((a_arr, hmcorr.af(a_arr))))
# [[0.9400534  0.19910454]
#  [0.88486195 0.21931514]
#  [0.84503291 0.23539276]
#  [0.81073196 0.25041556]
#  [0.77856825 0.26570189]
#  [0.74659465 0.28224283]]

# plot
plt.figure()
_ = [plt.axvline(a, c="k") for a in a_arr]
plt.plot(a_arr, hmcorr.k0f(a_arr), lw=3, label="$k_0$")
plt.plot(a_arr, hmcorr.sf(a_arr), lw=3, label="$\\sigma$")
plt.plot(a_arr, hmcorr.af(a_arr), lw=3, label="$A$")
plt.xlabel("$a$", fontsize=18)
plt.legend(loc="center", ncol=3, fancybox=True, fontsize=16)
plt.tight_layout()
plt.savefig("images/aHM_kHM_sHM.pdf", bbox_inches="tight")
