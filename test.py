"""
Quick test of tomographic (gg), (gy), (gk) using yxg best-fit parameters.
"""


import numpy as np
import matplotlib.pyplot as plt
from model.profile2D import Arnaud, HOD, Lensing
from model.power_spectrum import hm_ang_power_spectrum
from model.hmcorr import HaloModCorrection

outdir = "output_default/"

files = ["2mpz_2mpz", "2mpz_y_milca", "2mpz_lens",
         "wisc1_wisc1", "wisc1_y_milca", "wisc1_lens",
         "wisc2_wisc2", "wisc2_y_milca", "wisc2_lens",
         "wisc3_wisc3", "wisc3_y_milca", "wisc3_lens",
         "wisc4_wisc4", "wisc4_y_milca", "wisc4_lens",
         "wisc5_wisc5", "wisc5_y_milca", "wisc5_lens",]





ells = np.arange(2, 2500)

cosmo_args = {"Omega_c"       : 0.26066676,
              "Omega_b"       : 0.048974682,
              "h"             : 0.6766,
              "n_s"           : 0.9665,
              "sigma8"        : 0.8102,
              "mass_function" : "tinker"}

hm_corr = HaloModCorrection

kwargs = [{"M0":  11.60,
          "M1":  12.93,
          "Mmin":  12.01,
          "alpha":  1.0,
          "b_hydro":  0.6628,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  0.127,
          "sigma_lnM":  0.15,
          "width": 1.200,
          "a_HMcorr": 0.36},

          {"M0":  12.01,
          "M1":  13.37,
          "Mmin":  12.01,
          "alpha":  1.0,
          "b_hydro":  0.4715,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  -0.4999,
          "sigma_lnM":  0.15,
          "width": 1.144,
          "a_HMcorr": 0.36},


         {"M0":  12.01,
          "M1":  13.30,
          "Mmin":  12.01,
          "alpha":  1.0,
          "b_hydro":  0.4867,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  -0.5008,
          "sigma_lnM":  0.15,
          "width": 1.157,
          "a_HMcorr": 0.36},

          {"M0":  11.99,
          "M1":  13.23,
          "Mmin":  11.99,
          "alpha":  1.0,
          "b_hydro":  0.4656,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  -0.5492,
          "sigma_lnM":  0.15,
          "width": 1.194,
          "a_HMcorr": 0.36},

          {"M0":  12.02,
          "M1":  13.21,
          "Mmin":  12.02,
          "alpha":  1.0,
          "b_hydro":  0.4889,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  -0.5404,
          "sigma_lnM":  0.15,
          "width": 1.170,
          "a_HMcorr": 0.36},

          {"M0":  12.00,
          "M1":  12.77,
          "Mmin":  12.77,
          "alpha":  1.0,
          "b_hydro":  0.3920,
          "beta_gal":  1.0,
          "beta_max":  1.0,
          "fc":  1.0,
          "r_corr":  -0.6747,
          "sigma_lnM":  0.15,
          "width": 0.9533,
          "a_HMcorr": 0.36}

          ]

dndz = ["data/dndz/2MPZ_bin1.txt",
        "data/dndz/WISC_bin1.txt",
        "data/dndz/WISC_bin2.txt",
        "data/dndz/WISC_bin3.txt",
        "data/dndz/WISC_bin4.txt",
        "data/dndz/WISC_bin5.txt"]

col = ["r", "g", "b"]

prof_y = Arnaud()
prof_k = Lensing()


fig, ax = plt.subplots(6, 1, sharex=True, figsize=(7, 12))

for i, xx in enumerate(ax):
    prof_g = HOD(nz_file=dndz[i])

    for n in range(3):
        profiles = (prof_g, [prof_g, prof_y, prof_k][n])

        fname = files[3*i+n]
        with np.load(outdir+"cls_"+fname+".npz") as f:
            xx.loglog(f["ls"], f["cls"], "o", c=col[n], label=fname)

        kw = {**cosmo_args, **kwargs[i]}
        Cl = hm_ang_power_spectrum(ells, profiles, hm_correction=None, **kw)
        xx.loglog(ells, Cl, "k-", lw=2)

    xx.legend(loc="lower left", fontsize=8, ncol=3)
