"""
Unit test for number of sampling points in k (wavenumber)
and a (scale factor) in model/hm_ang_power_spectrum.
Overall we recuce runtime of hm_ang_power_spectrum by ~83%,
keeping an accuracy of well below 0.1%.
"""
import numpy as np
from analysis.params import ParamRun
import pyccl as ccl
from model.data import ProfTracer
from model.utils import get_hmcalc
from model.power_spectrum import hm_ang_power_spectrum


fname = "params_wisc1.yml"
p = ParamRun(fname)

for mg in p.get("maps"):
    if mg["name"] == "wisc1":
        g = ProfTracer(mg)
        break


for my in p.get("maps"):
    if my["name"] == "y_milca":
        y = ProfTracer(my)
        break


for mk in p.get("maps"):
    if mk["name"] == "lens":
        k = ProfTracer(mk)
        break


kwargs = mg["model"]
kwargs["mass_function"] = ccl.halos.hmfunc.mass_function_from_name("tinker08")
kwargs["halo_bias"] = ccl.halos.hbias.halo_bias_from_name("tinker10")

cosmo = p.get_cosmo()
hmc = get_hmcalc(cosmo, **kwargs)
l = np.arange(6, 2500)

clgg_256x64 = hm_ang_power_spectrum(cosmo, hmc, l, (g,g), kpts=256, zpts=64, **kwargs)
clgg_128x8 = hm_ang_power_spectrum(cosmo, hmc, l, (g,g), kpts=128, zpts=8, **kwargs)
assert (np.fabs(1-clgg_128x8/clgg_256x64) < 0.001).all()

clgy_256x64 = hm_ang_power_spectrum(cosmo, hmc, l, (g,y), kpts=256, zpts=64, **kwargs)
clgy_128x8 = hm_ang_power_spectrum(cosmo, hmc, l, (g,y), kpts=128, zpts=8, **kwargs)
assert (np.fabs(1-clgy_128x8/clgy_256x64) < 0.001).all()

clgk_256x64 = hm_ang_power_spectrum(cosmo, hmc, l, (g,k), kpts=256, zpts=64, **kwargs)
clgk_128x8 = hm_ang_power_spectrum(cosmo, hmc, l, (g,k), kpts=128, zpts=8, **kwargs)
assert (np.fabs(1-clgk_128x8/clgk_256x64) < 0.001).all()


"""
# bechmarks (k,a)::(g,g)
(256, 64) -> 1810  [err = 0.0000, % save --]

(128, 64) -> 1170  [err = 0.0002, % save 35]
(128, 8)  ->  359  [err = 0.0002, % save 80]  !!!
(128k, 8) ->  340  [err = 0.0001, % save 81]
(128k, 8a)->  338  [err = 0.0002, % save 81]  FINAL
(128, 4)  ->  277  [err = 0.0120, % save 85]

(64, 64)  ->  822  [err = 0.0165, % save 55]
(64, 32)  ->  505  [err = 0.0165, % save 72]
(64, 16)  ->  384  [err = 0.0164, % save 79]
(64, 8)   ->  310  [err = 0.0164, % save 83]
(64, 4)   ->  245  [err = 0.0164, % save 86]

(32, 64)  ->  625  [err = 0.0501, % save 65]


# bechmarks (k,a)::(g,y)
(256, 64) -> 1750  [err = 0.0000, % save --]

(128, 64) -> 1090  [err = 0.0005, % save 38]
(128, 8)  ->  280  [err = 0.0008, % save 84]  !!!
(128k, 8) ->  309  [err = 0.0008, % save 82]
(128k, 8a)->  290  [err = 0.0001, % save 83]  FINAL
(128, 4)  ->  223  [err = 0.0112, % save 87]

(64, 64)  ->  333  [err = 0.0091, % save 81]  !!
(64, 8)   ->  235  [err = 0.1112, % save 87]


# bechmarks (k,a)::(g,k)
(256, 64) -> 2280  [err = 0.0000, % save --]

(128, 64) -> 1290  [err = 0.0001, % save 43]
(128, 8)  ->  322  [err = 0.0002, % save 86]  !!!
(128k, 8) ->  336  [err = 0.0001, % save 85]
(128k, 8a)->  337  [err = 0.0001, % save 85]  FINAL
(128, 4)  ->  252  [err = 0.0112, % save 89]

(64, 64)  ->  859  [err = 0.0112, % save 62]
"""