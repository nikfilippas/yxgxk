"""
Unit test for Gaussian approximation of the halo model correction.
The approximation agrees with HaloFit within 5.5%.
"""

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import pyccl as ccl
from pyccl.halos.hmfunc import mass_function_from_name
from pyccl.halos.hbias import halo_bias_from_name
from model.hmcorr import HaloModCorrection
from model.hmcorr import HM_halofit


k_arr = np.geomspace(0.1, 2, 1000)   # wave number
a_arr = 1/(1+np.linspace(0, 1, 16))  # scale factor

cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)
kwargs = {"mass_function": mass_function_from_name("tinker08"),
          "halo_bias": halo_bias_from_name("tinker10")}

HM = HaloModCorrection(cosmo, **kwargs)
a_HMcorr = HM.af(a_arr)
for i, a in enumerate(a_arr):
    hm_old = HM_halofit(cosmo, **kwargs).rk_interp(k_arr, a)
    kwargs["a_HMcorr"] = a_HMcorr[i]
    hm_new = HM.hm_correction(k_arr, a, **kwargs)
    assert (np.fabs(1-hm_new/hm_old) < 0.053).all()
