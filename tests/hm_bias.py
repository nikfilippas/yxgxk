"""
Testing of the Halo Model bias module.
Old and new pipelines agree within 0.05%.
"""

import numpy as np
import pyccl as ccl
from analysis.params import ParamRun
from model.cosmo_utils import COSMO_DEFAULT
from model.data import ProfTracer
from model.power_spectrum import hm_bias
fname = "../../yxg/params_dam_wnarrow.yml"
p = ParamRun(fname)
cosmo = COSMO_DEFAULT()
z_arr = np.array([0.07, 0.13, 0.18, 0.23, 0.27, 0.32])
a_arr = 1/(1+z_arr)
hmfunc = ccl.halos.hmfunc.mass_function_from_name("tinker08")
hbias = ccl.halos.hbias.halo_bias_from_name("tinker10")
for m in p.get("maps"):
    if m["name"] == "y_milca":
        break
y = ProfTracer(m)
by = []
for a, m in zip(a_arr, p.get("maps")[:6]):
    kw = {**m["model"], **{"mass_function": hmfunc, "halo_bias":hbias}}
    y.update_parameters(cosmo, **kw)
    b = hm_bias(cosmo, a, y, **kw)
    by.append(b)
by = np.hstack((by))
bPe = np.load("bPe.npz")["bPe"]
assert (np.fabs(1-bPe/by) < 0.0005).all()

# Code to reproduce yxg hm_bias
# import numpy as np
# from analysis.params import ParamRun
# from model.power_spectrum import hm_bias
# from model.profile2D import Arnaud
# fname = "params_dam_wnarrow.yml"
# p = ParamRun(fname)
# cosmo = p.get_cosmo()
# z_arr = np.array([0.07, 0.13, 0.18, 0.23, 0.27, 0.32])
# a_arr = 1/(1+z_arr)
# y = Arnaud()
# bPe = [hm_bias(cosmo, a, y, **m["model"]) for a, m in zip(a_arr, p.get("maps")[:6])]
# bPe = np.hstack((bPe))
# np.savez("bPe", a_arr=a_arr, bPe=bPe)
