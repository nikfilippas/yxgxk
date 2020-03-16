"""compares yxg to yxgxk"""

## GLOBAL PARAMETERS ##
import numpy as np
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665,
                      mass_function="tinker")

cosmo_pars = {'Omega_c': 0.26066676,
              'Omega_b': 0.048974682,
              'h': 0.6766,
              'sigma8': 0.8102,
              'n_s': 0.9665,
              'mass_function': 'tinker',
              'a_HMcorr': 0.35015861}

kwargs = [[] for i in range(6)]
kwargs[0] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 12.939399020269125, 'Mmin': 11.616598183948195, 'b_hydro': 0.7878221462263877, 'r_corr': 0.6762753732566117, 'width': 1.1981397977341839, 'M0': 11.616598183948195, 'beta_max': 1.0}
kwargs[1] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 12.49345861209658, 'Mmin': 11.199442645998566, 'b_hydro': 0.33056689852361776, 'r_corr': -0.5893218315793198, 'width': 0.8971416426137775, 'M0': 11.199442645998566, 'beta_max': 1.0}
kwargs[2] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 11.716163989135934, 'Mmin': 10.470958015134814, 'b_hydro': 0.31194600628571223, 'r_corr': -0.5880946385481943, 'width': 0.8084603154884547, 'M0': 10.470958015134814, 'beta_max': 1.0}
kwargs[3] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 12.589710102154282, 'Mmin': 11.400264012460116, 'b_hydro': 0.38506384781329783, 'r_corr': -0.5896259515254361, 'width': 0.9846392322801006, 'M0': 11.400264012460116, 'beta_max': 1.0}
kwargs[4] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 13.214355615667605, 'Mmin': 12.017735153580738, 'b_hydro': 0.48888609516834475, 'r_corr': -0.5408371832521891, 'width': 1.1702306148409665, 'M0': 12.017735153580738, 'beta_max': 1.0}
kwargs[5] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 13.395678417795455, 'Mmin': 12.521243817029093, 'b_hydro': 0.4769459165197276, 'r_corr': -0.6549627438749408, 'width': 1.1739952216972653, 'M0': 12.521243817029093, 'beta_max': 1.0}



## HM CORRECTION
# old
from model.hmcorr import HalomodCorrection_old
hm_correction = HalomodCorrection_old(cosmo).rk_interp
# new
from model.hmcorr import HaloModCorrection
# hm_correction = HaloModCorrection


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
        lst[j][i] = np.load("legacy/data/lst%d%d.npy" % (i,j))
        tt[j][i] = np.load("legacy/data/tt%d%d.npy" % (i,j))


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
from model.power_spectrum import hm_ang_power_spectrum
from model.profile2D import HOD, Arnaud, Lensing
import pipeline_utils as pu
g = []
g.append(HOD(nz_file="data/dndz/2MPZ_bin1.txt"))
for i in range(5):
    g.append(HOD(nz_file="data/dndz/WISC_bin%d.txt" % (i+1)))

z = np.linspace(0.0005, 6, 1000)
y = Arnaud()
k = Lensing()

# z_ranges = [[0.0005, 0.1515],
#             [0.0005, 0.2875],
#             [0.0005, 0.3585],
#             [0.0175, 0.4365],
#             [0.0325, 0.5165],
#             [0.0355, 0.6085]]

TT = [[[] for i in range(6)] for j in range(3)]
for i in range(6):
    # determine z-range
    nz = g[i].nzf(z)
    z_inrange = z[nz >= 0.005*np.amax(nz)]
    z_range = [z_inrange[0], z_inrange[-1]]
    # z_range = z_ranges[i]
    print(z_range)

    TT[0][i] = hm_ang_power_spectrum(L, (g[i], g[i]),
                                     hm_correction=hm_correction,
                                     z_range=z_range, zpoints=64,
                                     **{**kwargs[i], **cosmo_pars})
    TT[0][i] *= pu.Beam(("g", "g"), L, 2048)
    TT[1][i] = hm_ang_power_spectrum(L, (y, g[i]),
                                     hm_correction=hm_correction,
                                     z_range=[1e-6, 6], zpoints=64,
                                     **{**kwargs[i], **cosmo_pars})
    TT[1][i] *= pu.Beam(("g", "y"), L, 2048)
    TT[2][i] = hm_ang_power_spectrum(L, (k, g[i]),
                                     hm_correction=hm_correction,
                                     z_range=[1e-6, 6],
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
                cov = np.load("legacy/data/cov_comb_m_%s_%s_%s_%s.npz" % ((zbins[i],)*4))
            elif j == 1:
                cov = np.load("legacy/data/cov_comb_m_%s_%s_%s_%s.npz" % ((zbins[i],"y_milca")*2))

            ax[j, i].set_ylim(ylims_min[j], ylims_max[j])

            yerr = np.sqrt(np.diag(cov["cov"]))
            # old data
            ax[j, i].errorbar(lsd[j][i], dd[j][i], yerr,
                              fmt="ro", ms=5, alpha=0.2, label="old data")
            # old theory
            ax[j, i].plot(lst[j][i], tt[j][i], "k-", lw=1.3, label="old fit")

fig.tight_layout(h_pad=0, w_pad=0)



#ax[0].legend(loc="upper right")


fig.savefig("comparison.pdf")
plt.close()
