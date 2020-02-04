# '''
import numpy as np
import matplotlib.pyplot as plt

nn1 = "output_default/cls_2mpz_2mpz.npz"
nn2 = "output_default/cls_y_milca_2mpz.npz"

gxg = np.load(nn1)
yxg = np.load(nn2)

gxgxk = np.load("../yxg/" + nn1)
yxgxk = np.load("../yxg/" + nn2)

plt.figure()
plt.loglog()

plt.errorbar(gxg["ls"], gxg["cls"], gxg["nls"], fmt="ro", alpha=0.2)
plt.errorbar(gxgxk["ls"], gxgxk["cls"], gxgxk["nls"], fmt="bo", alpha=0.2)


plt.figure()
plt.loglog()

plt.errorbar(yxg["ls"], yxg["cls"], yxg["nls"], fmt="ro", alpha=0.2)
plt.errorbar(yxgxk["ls"], yxgxk["cls"], yxgxk["nls"], fmt="bo", alpha=0.2)
'''
'''
##############################################################################

import emcee
import numpy as np
import matplotlib.pyplot as plt

savedir = '../yxg/output_default/'
fname_chain = savedir + 'sampler_lmin10_kmax1_tinker08_ymilca_wnarrow_2mpz_chain.h5'

reader = emcee.backends.HDFBackend(fname_chain, read_only=True)
chain = reader.get_chain(flat=True)
probs = reader.get_log_prob(flat=True)


BF = chain[np.argmax(probs)]

cd ../yxg
from model.power_spectrum import hm_ang_power_spectrum
from model.profile2D import Arnaud
from model.profile2D import HOD
H = HOD(nz_file="data/dndz/2MPZ_bin1.txt")
cd ../yxgxk

from model.power_spectrum import hm_ang_power_spectrum
from model.profile2D import Arnaud
from model.profile2D import HOD
from analysis.params import ParamRun

p = ParamRun("params_lensing.yml")

y = Arnaud()
g = HOD(nz_file="data/dndz/2MPZ_bin1.txt")


l = np.geomspace(6, 3e3, 1000)


bf_pars = {"M1": BF[0],
          "Mmin": BF[1],
          "M0": BF[1],
          "alpha": 1.0,
          "b_hydro": BF[2],
          "beta_gal": 1.0,
          "beta_max": 1.0,
          "fc": 1.0,
          "r_corr": BF[3],
          "sigma_lnM": 0.15,
          "width": BF[4],
          "a_HMcorr": 0.3614146096356469
          }
cosmo_pars = p.get_cosmo_pars()
kwargs = {**bf_pars, **cosmo_pars}


Clyg_yxgxk = hm_ang_power_spectrum(l, (y, g), hm_correction=None, **kwargs)
Clgg_yxgxk = hm_ang_power_spectrum(l, (g, g), hm_correction=None, **kwargs)


# Clyg_yxgxk = hm_ang_power_spectrum(l, (y, g), hm_correction=hm_correction, **kwargs)
# Clgg_yxgxk = hm_ang_power_spectrum(l, (g, g), hm_correction=hm_correction, **kwargs)
# Cl_yxg = pspec1(l, (A1, H), **kwargs)