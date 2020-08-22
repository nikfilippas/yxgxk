"""
Testing compatibility of the implementation of the 4-pt function
in the old and new pipelines.
"""

import numpy as np
import pyccl as ccl
from model.trispectrum import hm_1h_trispectrum
from analysis.params import ParamRun
from model.cosmo_utils import COSMO_DEFAULT
from model.data import ProfTracer

fname = "../yxg/params_dam_wnarrow.yml"
p = ParamRun(fname)
cosmo = COSMO_DEFAULT()
k_arr = np.logspace(-3, 3, 100)
for m in p.get("maps"):
    if m["name"] == "wisc3":
        g = ProfTracer(m)
    if m["name"] == "y_milca":
        y = ProfTracer(m)
for m in p.get("maps"):
    if m["name"] == "wisc3":
        break
hmfunc = ccl.halos.hmfunc.mass_function_from_name("tinker08")
hbias = ccl.halos.hbias.halo_bias_from_name("tinker10")
kw = {**m["model"], **{"mass_function": hmfunc, "halo_bias":hbias}}
kw["lM0"] = kw["M0"]
kw["lM1"] = kw["M1"]
kw["lMmin"] = kw["Mmin"]
kw["r_corr_gy"] = kw["r_corr"]
g.update_parameters(cosmo, **kw)
y.update_parameters(cosmo, **kw)
cov_gggg = hm_1h_trispectrum(cosmo, k_arr, 0.23, (g,g,g,g), **kw)
cov_gggy = hm_1h_trispectrum(cosmo, k_arr, 0.23, (g,g,g,y), **kw)
cov_gygy = hm_1h_trispectrum(cosmo, k_arr, 0.23, (g,y,g,y), **kw)
# load previous trispectra
qcov = np.load("cov_1h4pt.npz")
qcov_gggg = qcov["cov_gggg"]
qcov_gggy = qcov["cov_gggy"]
qcov_gygy = qcov["cov_gygy"]

assert (np.fabs(1-cov_gggg/qcov_gggg) < 0.05).all()
assert (np.fabs(1-cov_gggy/qcov_gggy) < 0.05).all()
assert (np.fabs(1-cov_gygy/qcov_gygy) < 0.05).all()


# Code to reproduce old 1h4pt covs
# import numpy as np
# from model.trispectrum import hm_1h_trispectrum
# from analysis.params import ParamRun
# from model.profile2D import HOD, Arnaud
# fname = "params_dam_wnarrow.yml"
# p = ParamRun(fname)
# cosmo = p.get_cosmo()
# k_arr = np.logspace(-3, 3, 100)
# a_arr = 1/(1+0.23)
# y = Arnaud()
# g = HOD(nz_file="data/dndz/WISC_bin3.txt")
# for m in p.get("maps"):
#     if m["name"] == "wisc3":
#         break
# cov_gggg = hm_1h_trispectrum(cosmo, k_arr, a_arr, (g,g,g,g), **m["model"])
# cov_gggy = hm_1h_trispectrum(cosmo, k_arr, a_arr, (g,g,g,y), **m["model"])
# cov_gygy = hm_1h_trispectrum(cosmo, k_arr, a_arr, (g,y,g,y), **m["model"])
# np.savez("cov_1h4pt", a_arr=a_arr,
#                       cov_gggg=cov_gggg,
#                       cov_gggy=cov_gggy,
#                       cov_gygy=cov_gygy)
