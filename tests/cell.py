"""
Unit test for the CCL-based implementation of gxg and gxy.
Comparison with old code agrees within 0.1% for gxg and 1% for gxy.
"""

from analysis.params import ParamRun
from pyccl.halos.hmfunc import mass_function_from_name
from pyccl.halos.hbias import halo_bias_from_name
from model.data import ProfTracer
import numpy as np
from model.power_spectrum import hm_ang_power_spectrum
from model.cosmo_utils import COSMO_ARGS
fname = "../yxg/params_dam_wnarrow.yml"
p = ParamRun(fname)
zbin = "wisc3"
kwargs = p.get_models()[zbin]
kwargs['lMmin'] = kwargs['Mmin']
kwargs['lM0'] = kwargs['M0']
kwargs['lM1'] = kwargs['M1']
kwargs['sigmaLogM'] = kwargs['sigma_lnM']
kwargs["r_corr_gy"] = kwargs['r_corr']
if p.get_massfunc() == "tinker":
    kwargs['mass_function'] = mass_function_from_name("tinker08")
    kwargs['halo_bias'] = halo_bias_from_name("tinker10")

l = np.geomspace(6, 3000, 100)

# gxg
for m in p.get("maps"):
    if m["name"] == zbin:
        break
# scale cuts
kmax = p.get("mcmc")["kmax"]
p1 = p2 = ProfTracer(m, kmax)
# msk = np.ones(100).astype(int)
msk = tuple([l < p1.lmax])

cosmo = COSMO_ARGS(kwargs)
cl_new = hm_ang_power_spectrum(cosmo, l, (p1, p2), **kwargs)
cl_old = np.load("tests/cl_test_%s.npz" % zbin)["clgg"]
import matplotlib.pyplot as plt
plt.figure()
plt.loglog(l[msk], cl_old[msk])
plt.loglog(l[msk], cl_new[msk])
assert (np.fabs(1-cl_new[msk]/cl_old[msk]) < 0.01).all()

# gxy
for n in p.get("maps"):
    if n["name"] == "y_milca":
        break
p2 = ProfTracer(n)

cl_new = hm_ang_power_spectrum(cosmo, l, (p1, p2), **kwargs)
cl_old = np.load("tests/cl_test_%s.npz" % zbin)["clgy"]
plt.figure()
plt.loglog(l[msk], cl_old[msk])
plt.loglog(l[msk], cl_new[msk])
assert (np.fabs(1-cl_new[msk]/cl_old[msk]) < 0.01).all()


# Code to reproduce yxg cells
# import numpy as np
# from analysis.params import ParamRun
# from model.power_spectrum import hm_ang_power_spectrum
# from model.profile2D import HOD, Arnaud
# fname = "params_dam_wnarrow.yml"
# p = ParamRun(fname)
# zbin = "wisc3"  # change this if you want
# cosmo = p.get_cosmo()
# y = Arnaud()
# for m in p.get("maps"):
#     if m["name"] == zbin:
#         break
# nz_file = m["dndz"]
# z, nz = np.loadtxt(nz_file, unpack=True)
# nz_hi = np.where(nz > np.max(nz)/100)[0]
# zrange = [z[nz_hi[0]], z[nz_hi[-1]]]
# g = HOD(nz_file=nz_file)
# l = np.geomspace(6, 3000, 100)
# kwargs = p.get_models()[zbin]
# zpoints = 256
# clgg = hm_ang_power_spectrum(cosmo, l, (g,g),
#                               zrange=zrange,
#                               zpoints=zpoints,
#                               **kwargs)
# clgy = hm_ang_power_spectrum(cosmo, l, (g,y),
#                               zrange=zrange,
#                               zpoints=zpoints,
#                               **kwargs)
# np.savez("cl_test_%s.npz" % zbin, l=l, clgg=clgg, clgy=clgy)
