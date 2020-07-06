"""compares yxg to yxgxk"""

import numpy as np
from analysis.params import ParamRun
from model.hmcorr import HaloModCorrection
from model.hmcorr import HM_halofit


p = ParamRun("params_lensing.yml")
cosmo_pars = p.get_cosmo_pars()
cosmo = p.get_cosmo()
hm_correction = HaloModCorrection(cosmo, **cosmo_pars).hm_correction
# hm_correction = HM_halofit(cosmo, **cosmo_pars).rk_interp

kwargs = [[] for i in range(6)]
kwargs[0] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 12.99472018, 'lMmin': 11.66317664, 'b_hydro': 0.33652023, 'r_corr_gy': -0.10361521, 'width': 1.19939348, 'lM0': 11.66317664, 'beta_max': 1.0, 'a_HMcorr': 0.1958513}
kwargs[1] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 11.89126683, 'lMmin': 10.59153793, 'b_hydro': 0.11877272, 'r_corr_gy': -0.65621202, 'width': 0.81297445, 'lM0': 11.59153793, 'beta_max': 1.0, 'a_HMcorr': 0.21515478}
kwargs[2] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 11.73255203, 'lMmin': 10.48718201, 'b_hydro': 0.16081615, 'r_corr_gy': -0.61417099, 'width': 0.80801670, 'lM0': 10.48718201, 'beta_max': 1.0, 'a_HMcorr': 0.23087286}
kwargs[3] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 12.51536196, 'lMmin': 11.32914985, 'b_hydro': 0.19452851, 'r_corr_gy': -0.62590662, 'width': 0.96975127, 'lM0': 11.32914985, 'beta_max': 1.0, 'a_HMcorr': 0.24627099}
kwargs[4] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 13.18622206, 'lMmin': 11.99335452, 'b_hydro': 0.26434493, 'r_corr_gy': -0.57767130, 'width': 1.15909897, 'lM0': 11.99335452, 'beta_max': 1.0, 'a_HMcorr': 0.25827266}
kwargs[5] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigmaLogM': 0.15, 'lM1': 13.43386643, 'lMmin': 12.55077359, 'b_hydro': 0.25261573, 'r_corr_gy': -0.68876424, 'width': 1.18987421, 'lM0': 12.55077359, 'beta_max': 1.0, 'a_HMcorr': 0.27359295}


## PREVIOUS ANALYSIS ##
lsd = [[[] for i in range(6)] for j in range(2)]
dd = [[[] for i in range(6)] for j in range(2)]
lst = [[[] for i in range(6)] for j in range(2)]
tt = [[[] for i in range(6)] for j in range(2)]
for j in range(2):
    for i in range(6):
        # data
        lsd[j][i] = np.load("legacy/data/lsd%d%d.npy" % (i, j))
        dd[j][i] = np.load("legacy/data/dd%d%d.npy" % (i,j))
        # theory
        D = np.load("tests/data%d%d.npz" % (i,j))
        lst[j][i] = D["l"]
        tt[j][i] = D["cl"]


## NEW ANALYSIS ##
LSD = [[[] for i in range(6)] for j in range(3)]
# gxg data
LSD[0][0] = np.load("output_default/cls_2mpz_2mpz.npz")
LSD[0][1] = np.load("output_default/cls_wisc1_wisc1.npz")
LSD[0][2] = np.load("output_default/cls_wisc2_wisc2.npz")
LSD[0][3] = np.load("output_default/cls_wisc3_wisc3.npz")
LSD[0][4] = np.load("output_default/cls_wisc4_wisc4.npz")
LSD[0][5] = np.load("output_default/cls_wisc5_wisc5.npz")
# yxg data
LSD[1][0] = np.load("output_default/cls_2mpz_y_milca.npz")
LSD[1][1] = np.load("output_default/cls_wisc1_y_milca.npz")
LSD[1][2] = np.load("output_default/cls_wisc2_y_milca.npz")
LSD[1][3] = np.load("output_default/cls_wisc3_y_milca.npz")
LSD[1][4] = np.load("output_default/cls_wisc4_y_milca.npz")
LSD[1][5] = np.load("output_default/cls_wisc5_y_milca.npz")
# yxk data
LSD[2][0] = np.load("output_default/cls_2mpz_lens.npz")
LSD[2][1] = np.load("output_default/cls_wisc1_lens.npz")
LSD[2][2] = np.load("output_default/cls_wisc2_lens.npz")
LSD[2][3] = np.load("output_default/cls_wisc3_lens.npz")
LSD[2][4] = np.load("output_default/cls_wisc4_lens.npz")
LSD[2][5] = np.load("output_default/cls_wisc5_lens.npz")

# global theory
L = np.geomspace(6, 3000, 50)
from analysis.params import ParamRun
from model.power_spectrum import hm_ang_power_spectrum
from model.data import ProfTracer
import analysis.pipeline_utils as pu

g = []
for m in p.get("maps"):
    if ("2mpz" in m["name"]) or ("wisc" in m["name"]):
        g.append(ProfTracer(m))
    if "y_milca" in m["name"]:
        y = ProfTracer(m)
    if "lens" in m["name"]:
        k = ProfTracer(m)


z = np.linspace(0.0005, 6, 1000)


TT = [[[] for i in range(6)] for j in range(3)]
for i in range(6):
    TT[0][i] = hm_ang_power_spectrum(L, (g[i], g[i]),
                                     hm_correction=hm_correction,
                                     **{**kwargs[i], **cosmo_pars})
    TT[0][i] *= pu.Beam(("g", "g"), L, 2048)
    TT[1][i] = hm_ang_power_spectrum(L, (y, g[i]),
                                     hm_correction=hm_correction,
                                     **{**kwargs[i], **cosmo_pars})
    TT[1][i] *= pu.Beam(("g", "y"), L, 2048)
    TT[2][i] = hm_ang_power_spectrum(L, (k, g[i]),
                                     hm_correction=hm_correction,
                                     **{**kwargs[i], **cosmo_pars})
    TT[2][i] *= pu.Beam(("g", "k"), L, 2048)



## FIGURE ###
zbins = ["2mpz"] + ["wisc%d" % d for d in range(1, 6)]
corrs = [r"$g \times g$", r"$g \times y$", r"$g \times \kappa$"]
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 6, sharex=True, sharey="row", figsize=(15,6))

[a.set_ylabel(corr, fontsize=15, labelpad=15) for a, corr in zip(ax[:, 0], corrs)]
[a.set_xlabel(zbin, fontsize=15, labelpad=15) for a, zbin in zip(ax[0], sci)]
[a.xaxis.set_label_position("top") for a in ax[0]]
[a.set_xlabel(r"$\ell$", fontsize=14) for a in ax[-1]]

ylims_min = np.min([[np.min(x[x > 0]) for x in dd[i]] for i in range(2)], axis=1)
ylims_max = np.max([[np.max(x) for x in dd[i]] for i in range(2)], axis=1)

for j in range(3):
    for i in range(6):
        ax[j, i].loglog()

        try:
            if j == 0:
                cov = np.load("output_default/cov_data_%s_%s_%s_%s.npz" % ((zbins[i],)*4))
            elif j == 1:
                cov = np.load("output_default/cov_data_%s_%s_%s_%s.npz" % ((zbins[i],"y_milca")*2))
            elif j == 2:
                cov = np.load("output_default/cov_data_%s_%s_%s_%s.npz" % ((zbins[i],"lens")*2))

            yerr = np.sqrt(np.diag(cov["cov"]))
        except FileNotFoundError:
            yerr = np.zeros_like(LSD[j][i]["ls"])

        # new data
        ax[j, i].errorbar(LSD[j][i]["ls"], LSD[j][i]["cls"]-LSD[j][i]["nls"], yerr,
                          fmt="bo", ms=4, alpha=0.2, label="new data")
        # new theory
        ax[j, i].plot(L, TT[j][i], "g-", lw=1, label="new fit")

        if j < 2:
            if j == 0:
                cov = np.load("../yxg/output_default/cov_comb_m_%s_%s_%s_%s.npz" % ((zbins[i],)*4))
            elif j == 1:
                cov = np.load("../yxg/output_default/cov_comb_m_%s_%s_%s_%s.npz" % ((zbins[i],"y_milca")*2))

            ax[j, i].set_ylim(ylims_min[j], ylims_max[j])

            yerr = np.sqrt(np.diag(cov["cov"]))
            # old data
            ax[j, i].errorbar(lsd[j][i], dd[j][i], yerr,
                              fmt="ro", ms=5, alpha=0.2, label="old data")
            # old theory
            ax[j, i].plot(lst[j][i], tt[j][i], "k-", lw=1.3, label="old fit")

fig.tight_layout(h_pad=0, w_pad=0)
fig.savefig("comparison_cell.pdf")
# plt.close()
