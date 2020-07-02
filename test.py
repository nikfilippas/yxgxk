# fname = "params_wnarrow.yml"
# from analysis.params import ParamRun
# p = ParamRun(fname)
# cosmo = p.get_cosmo()
# from model.power_spectrum import hm_ang_power_spectrum
# from model.profile2D import HOD
# g = HOD(nz_file="data/dndz/WISC_bin3.txt")
# import numpy as np
# l = np.geomspace(6, 3000, 100)
# kwargs = p.get_models()["wisc3"]
# zrange = [0.0175, 0.4365]
# zpoints = 64
# cl = hm_ang_power_spectrum(cosmo, l, (g,g),
#                             zrange=zrange,
#                             zpoints=zpoints,
#                             **kwargs)


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
#                            zrange=zrange,
#                            zpoints=zpoints,
#                            **kwargs)
# np.savez("cl_test.npz", l=l, cl=cl)
############################################

from analysis.params import ParamRun
import pyccl as ccl
from model.data import ProfTracer
import numpy as np
from model.power_spectrum import hm_ang_power_spectrum
fname = "../yxg/params_wnarrow.yml"
p = ParamRun(fname)
# default cosmology copied from yxg/analysis/params.ParamRun.get_cosmo
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665,
                      mass_function="tinker")
m = p.get("maps")[3]
p1 = p2 = ProfTracer(m)
kwargs = p.get_models()["wisc3"]
kwargs['lMmin'] = kwargs['Mmin']
kwargs['lM0'] = kwargs['M0']
kwargs['lM1'] = kwargs['M1']
kwargs['sigmaLogM'] = kwargs['sigma_lnM']
kwargs['width'] = 1
l = np.geomspace(6, 3000, 100)
cl = hm_ang_power_spectrum(l, (p1, p2), **kwargs)
np.savez('cl_test_1.npz', l=l, cl=cl)


# fname = "../yxg/params_wnarrow.yml"
# from analysis.params import ParamRun
# p = ParamRun(fname)
# import pyccl as ccl
# # default cosmology copied from yxg/analysis/params.ParamRun.get_cosmo
# cosmo = ccl.Cosmology(Omega_c=0.26066676,
#                              Omega_b=0.048974682,
#                              h=0.6766,
#                              sigma8=0.8102,
#                              n_s=0.9665,
#                              mass_function="tinker")
# from model.data import ProfTracer
# m = p.get("maps")[3]
# p1 = p2 = ProfTracer(m)
# kwargs = p.get_models()["wisc3"]
# kwargs["lMmin"] = kwargs["Mmin"]
# kwargs["lM0"] = kwargs["M0"]
# kwargs["lM1"] = kwargs["M1"]
# kwargs["sigmaLogM"] = kwargs["sigma_lnM"]
# import numpy as np
# l = np.geomspace(6, 3000, 100)
# from model.power_spectrum import hm_ang_power_spectrum
# cl = hm_ang_power_spectrum(l, (p1, p2), **kwargs)
