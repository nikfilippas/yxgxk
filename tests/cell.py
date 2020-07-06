"""
Unit test for the CCL-based implementation of gxg.
Comparison with old code agrees at a level better than 1%.
"""

from analysis.params import ParamRun
import pyccl as cc
from pyccl.halos.hmfunc import mass_function_from_name
from pyccl.halos.hbias import halo_bias_from_name
from model.data import ProfTracer
import numpy as np
from model.power_spectrum import hm_ang_power_spectrum
fname = "../../yxg/params_wnarrow.yml"
p = ParamRun(fname)
m = p.get("maps")[3]
m["dndz"] = "../"+m["dndz"]

p1 = p2 = ProfTracer(m)
kwargs = p.get_models()["wisc3"]
kwargs['lMmin'] = kwargs['Mmin']
kwargs['lM0'] = kwargs['M0']
kwargs['lM1'] = kwargs['M1']
kwargs['sigmaLogM'] = kwargs['sigma_lnM']
kwargs['width'] = 1
if p.get_massfunc() == "tinker":
    kwargs['mass_function'] = mass_function_from_name("tinker08")
    kwargs['halo_bias'] = halo_bias_from_name("tinker10")
l = np.geomspace(6, 3000, 100)
cl_new = hm_ang_power_spectrum(l, (p1, p2), **kwargs)

cl_old = np.load("cl_test.npz")["cl"]
assert (np.fabs(1-cl_new/cl_old) < 0.001).all()


# # Code to produce yxg cells
# import numpy as np
# from analysis.params import ParamRun
# from model.power_spectrum import hm_ang_power_spectrum
# from model.profile2D import HOD
# fname = "params_wnarrow.yml"
# p = ParamRun(fname)
# cosmo = p.get_cosmo()
# g = HOD(nz_file="data/dndz/WISC_bin3.txt")
# l = np.geomspace(6, 3000, 100)
# kwargs = p.get_models()["wisc3"]
# kwargs['width'] = 1.
# zrange = [0.001, 0.6]
# zpoints = 256
# cl = hm_ang_power_spectrum(cosmo, l, (g,g),
#                             zrange=zrange,
#                             zpoints=zpoints,
#                             **kwargs)
# np.savez("cl_test.npz", l=l, cl=cl)
