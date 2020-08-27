"""
Testing compatibility of the implementation of the 4-pt function
in the old and new pipelines.
Old and New trispectra are within the same order of magnitude.
"""

import numpy as np
import pyccl as ccl
from model.trispectrum import hm_ang_1h_covariance
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
fsky = 0.6
l_arr = np.geomspace(6, 2500, 50)
zrange = [1e-6, 0.45]
cov_gggg = hm_ang_1h_covariance(fsky, l_arr, cosmo, (g,g,g,g),
                                zrange=zrange,
                                **kw)
cov_gggy = hm_ang_1h_covariance(fsky, l_arr, cosmo, (g,g,g,y),
                                zrange=zrange,
                                **kw)
cov_gygy = hm_ang_1h_covariance(fsky, l_arr, cosmo, (g,y,g,y),
                                zrange=zrange,
                                **kw)

# load previous trispectra
qcov = np.load("cov_1h4pt.npz")
qcov_gggg = qcov["cov_gggg"]
qcov_gggy = qcov["cov_gggy"]
qcov_gygy = qcov["cov_gygy"]

assert (np.fabs(np.log10(cov_gggg/qcov_gggg)) < 1.0).all()
assert (np.fabs(np.log10(cov_gggy/qcov_gggy)) < 1.1).all()
assert (np.fabs(np.log10(cov_gygy/qcov_gygy)) < 1.0).all()


# Code to reproduce old 1h4pt covs
# import numpy as np
# from model.trispectrum import hm_ang_1h_covariance
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
# fsky = 0.6
# l_arr = np.geomspace(6, 2500, 50)
# zrange_a = zrange_b = [1e-6, 0.45]
# cov_gggg = hm_ang_1h_covariance(cosmo, fsky, l_arr,
#                                 (g,g), (g,g),
#                                 zrange_a=zrange_a,
#                                 zrange_b=zrange_b,
#                                 **m["model"])
# cov_gggy = hm_ang_1h_covariance(cosmo, fsky, l_arr,
#                                 (g,g), (g,y),
#                                 zrange_a=zrange_a,
#                                 zrange_b=zrange_b,
#                                 **m["model"])
# cov_gygy = hm_ang_1h_covariance(cosmo, fsky, l_arr,
#                                 (g,y), (g,y),
#                                 zrange_a=zrange_a,
#                                 zrange_b=zrange_b,
#                                 **m["model"])
# np.savez("cov_1h4pt", cov_gggg=cov_gggg,
#                       cov_gggy=cov_gggy,
#                       cov_gygy=cov_gygy)
