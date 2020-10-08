"""
Comparison of different halo model correction models.
"""

import numpy as np
import pyccl as ccl
from pyccl.halos.halo_model import halomod_power_spectrum
from analysis.params import ParamRun
from model.utils import get_hmcalc
from model.data import ProfTracer
from model.power_spectrum import get_2pt


fname = "params_wisc1.yml"
p = ParamRun(fname)
cosmo = p.get_cosmo()
for m in p.get("maps"):
    if m["name"] == "wisc1":
        break
kwargs = m["model"]
kwargs["mass_function"] = ccl.halos.hmfunc.mass_function_from_name("tinker08")
kwargs["halo_bias"] = ccl.halos.hbias.halo_bias_from_name("tinker10")
hmc = get_hmcalc(cosmo, **kwargs)
l = np.arange(6, 2500)
g = ProfTracer(m)
kpts = 128
zpts = 8
p2pt = get_2pt(g, g, **kwargs)
k_arr = np.geomspace(1e-3, 1e2, kpts)
a_arr = np.linspace(1/(1+0.5), 1, zpts)

def pspec(hmc, prof=None, p2pt=None, get_1h=True, get_2h=True, f_ka=None, alpha=1.):
    return halomod_power_spectrum(cosmo, hmc,
                                  k_arr, g.z_avg,
                                  prof=prof,
                                  prof_2pt=p2pt,
                                  normprof1=True,
                                  normprof2=True,
                                  get_1h=get_1h,
                                  get_2h=get_2h,
                                  f_ka=f_ka,
                                  alpha=alpha)

Pgg_1h = pspec(hmc, g.profile, p2pt, True, False)
Pgg_2h = pspec(hmc, g.profile, p2pt, False, True)
Pgg = Pgg_1h + Pgg_2h

# HALOFIT 500c
from model.hmcorr import HM_halofit
hmcorr = HM_halofit(cosmo, Delta=500, rho_type="critical", **kwargs)
hf_500c = hmcorr.rk_interp(k_arr, 1/(1+g.z_avg), **kwargs)

# HALOFIT 200m
hmcorr = HM_halofit(cosmo, Delta=200, rho_type="matter", **kwargs)
hf_200m = hmcorr.rk_interp(k_arr, 1/(1+g.z_avg), **kwargs)

# Gauss
from model.hmcorr import HM_Gauss
hmcorr = HM_Gauss(cosmo, **kwargs)
kw = {"a_HMcorr": hmcorr.af(1/(1+g.z_avg))}
gauss = hmcorr.hm_correction(k_arr, 1/(1+g.z_avg), **kw)

# Mead
a = np.linspace(0.5, 1.0, 10)
P_mead = [pspec(hmc, g.profile, p2pt, alpha=aa) for aa in a]
mead = [Pm/(Pgg_1h+Pgg_2h) for Pm in P_mead]

import matplotlib.pyplot as plt
plt.figure()
plt.semilogx(k_arr, hf_200m, "k", lw=3, label="HALOFIT 200m")
plt.semilogx(k_arr, gauss, "grey", ls="--", lw=3, label="HF200m-calibrated Gauss")
plt.semilogx(k_arr, hf_500c, "b", label="HALOFIT 500c")
plt.semilogx(k_arr, mead[-5], "r", label="Mead et al, a=[1.0, 0.5]")
[plt.semilogx(k_arr, mm, "r", lw=0.7) for mm in mead]
plt.ylim(0.97, 1.4)
plt.xlabel("k")
plt.ylabel("hm_corr / (1-halo + 2-halo)")
plt.legend(loc="upper right")
