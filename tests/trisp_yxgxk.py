import numpy as np
from analysis.params import ParamRun
from model.data import ProfTracer
from model.utils import get_hmcalc
from model.power_spectrum import get_2pt
from model.trispectrum import hm_1h_trispectrum

fname = "params_lensing.yml"
p = ParamRun(fname)
cosmo = p.get_cosmo()

for m in p.get("maps"):
    if m["name"] == "wisc3":
        mg = m
    elif m["name"] == "y_milca":
        my = m
        break

kwargs = {**mg["model"], **p.get_cosmo_pars()}
hmc = get_hmcalc(cosmo, **kwargs)


g = ProfTracer(mg)
g.update_parameters(cosmo, **kwargs)
y = ProfTracer(my)
y.update_parameters(cosmo, **kwargs)

k = np.logspace(-3, 2, 256)
a = 1/(1+g.z_avg)

p2pt_gg = get_2pt(g, g, **kwargs)
p2pt_gy = get_2pt(g, y, **kwargs)
p2pt_yy = get_2pt(y, y, **kwargs)

tri_gggg = hm_1h_trispectrum(cosmo, hmc, k, a, (g,g,g,g),
                             p2pt_gg, p2pt_gg, **kwargs)
tri_gggy = hm_1h_trispectrum(cosmo, hmc, k, a, (g,g,g,y),
                             p2pt_gg, p2pt_gy, **kwargs)
tri_gygy = hm_1h_trispectrum(cosmo, hmc, k, a, (g,y,g,y),
                             p2pt_gy, p2pt_gy, **kwargs)
tri_yyyy = hm_1h_trispectrum(cosmo, hmc, k, a, (y,y,y,y),
                             p2pt_yy, p2pt_yy, **kwargs)

tri = np.load("tri.npz")

assert np.allclose(tri_gggg, tri["tri_gggg"], rtol=0.02)
assert np.allclose(tri_gggy, tri["tri_gggy"], rtol=0.02)
assert np.allclose(tri_gygy, tri["tri_gygy"], rtol=0.01)
assert np.allclose(tri_yyyy, tri["tri_yyyy"], rtol=0.01)

### TRISPECTRUM COVARIANCES ###
from model.trispectrum import hm_ang_1h_covariance
fsky = 0.6
l = np.arange(6, 2500, 10)

cov_gggg = hm_ang_1h_covariance(fsky, l, cosmo, hmc, (g,g,g,g),
                                p2pt_gg, p2pt_gg, **kwargs)
cov_gggy = hm_ang_1h_covariance(fsky, l, cosmo, hmc, (g,g,g,y),
                                p2pt_gg, p2pt_gy, **kwargs)
cov_gygy = hm_ang_1h_covariance(fsky, l, cosmo, hmc, (g,y,g,y),
                                p2pt_gy, p2pt_gy, **kwargs)
cov_yyyy = hm_ang_1h_covariance(fsky, l, cosmo, hmc, (y,y,y,y),
                                p2pt_yy, p2pt_yy, **kwargs)

cov = np.load("dcov.npz")

assert np.allclose(cov_gggg, cov["cov_gggg"], rtol=0.01)
assert np.allclose(cov_gggy, cov["cov_gggy"], rtol=0.01)
assert np.allclose(cov_gygy, cov["cov_gygy"], rtol=0.01)
assert np.allclose(cov_yyyy, cov["cov_yyyy"], rtol=0.01)


'''
#### yxg reproduction code ####
import numpy as np
from analysis.params import ParamRun
from model.profile2D import HOD, Arnaud
from model.trispectrum import hm_1h_trispectrum

fname = "params_lensing.yml"
p = ParamRun(fname)
p.p["mcmc"]["mfunc"] = "tinker"
cosmo = p.get_cosmo()

for m in p.get("maps"):
    if m["name"] == "wisc3":
        mg = m
    elif m["name"] == "y_milca":
        my = m
        break

g = HOD(nz_file=mg["dndz"])
y = Arnaud()

k = np.logspace(-3, 2, 256)
a = 1/(1+g.z_avg)

kwargs = mg["model"]
kwargs["Mmin"] = kwargs["lMmin"]
kwargs["M0"] = kwargs["lM0"]
kwargs["M1"] = kwargs["lM1"]
kwargs["sigma_lnM"] = kwargs["sigmaLogM"]
kwargs["r_corr"] = kwargs["r_corr_gy"]

tri_gggg = hm_1h_trispectrum(cosmo, k, a, (g,g,g,g), **kwargs)
tri_gggy = hm_1h_trispectrum(cosmo, k, a, (g,g,g,y), **kwargs)
tri_gygy = hm_1h_trispectrum(cosmo, k, a, (g,y,g,y), **kwargs)
tri_yyyy = hm_1h_trispectrum(cosmo, k, a, (y,y,y,y), **kwargs)

np.savez("../yxgxk/tri",
         tri_gggg=tri_gggg,
         tri_gggy=tri_gggy,
         tri_gygy=tri_gygy,
         tri_yyyy=tri_yyyy)


### TRISPECTRUM COVARIANCES ###
from model.trispectrum import hm_ang_1h_covariance


fsky = 0.6
l = np.arange(6, 2500, 10)
zrange_g = np.array([5e-04, 5.985e-01])
zrange_y = np.array([1e-6, 6])

cov_gggg = hm_ang_1h_covariance(cosmo, fsky, l, (g,g), (g,g),
                                zrange_a=zrange_g, zpoints_a=64,
                                zrange_b=zrange_g, zpoints_b=64,
                                **kwargs)
cov_gggy = hm_ang_1h_covariance(cosmo, fsky, l, (g,g), (g,y),
                                zrange_a=zrange_g, zpoints_a=64,
                                zrange_b=zrange_g, zpoints_b=64,
                                **kwargs)
cov_gygy = hm_ang_1h_covariance(cosmo, fsky, l, (g,y), (g,y),
                                zrange_a=zrange_g, zpoints_a=64,
                                zrange_b=zrange_g, zpoints_b=64,
                                **kwargs)
cov_yyyy = hm_ang_1h_covariance(cosmo, fsky, l, (y,y), (y,y),
                                zrange_a=zrange_y, zpoints_a=64,
                                zrange_b=zrange_y, zpoints_b=64,
                                **kwargs)

np.savez("../yxgxk/dcov",
         cov_gggg=cov_gggg,
         cov_gggy=cov_gggy,
         cov_gygy=cov_gygy,
         cov_yyyy=cov_yyyy)
'''

# mean and std of offsets (uncomment to run in this script)
# def test(a1, a2):
#     m = np.mean(1-a1/a2)
#     s = np.std(1-a1/a2)
#     print(m, s, np.max(np.abs(1-a1/a2)))

# test(tri_gggg, tri["tri_gggg"])
# test(tri_gggy, tri["tri_gggy"])
# test(tri_gygy, tri["tri_gygy"])
# test(tri_yyyy, tri["tri_yyyy"])

# test(cov_gggg, cov["cov_gggg"])
# test(cov_gggy, cov["cov_gggy"])
# test(cov_gygy, cov["cov_gygy"])
# test(cov_yyyy, cov["cov_yyyy"])
