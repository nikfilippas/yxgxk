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

kwargs[0] = {'fc': 1.0,
             'alpha': 1.0,
             'beta_gal': 1.0,
             'sigma_lnM': 0.15,
             'M1': 12.939399020269125,
             'Mmin': 11.616598183948195,
             'b_hydro': 0.7878221462263877,
             'r_corr': 0.6762753732566117,
             'width': 1.1981397977341839,
             'M0': 11.616598183948195,
             'beta_max': 1.0}

kwargs[1] = {'fc': 1.0, 'alpha': 1.0, 'beta_gal': 1.0, 'sigma_lnM': 0.15, 'M1': 12.939399020269125, 'Mmin': 11.616598183948195, 'b_hydro': 0.7878221462263877, 'r_corr': 0.6762753732566117, 'width': 1.1981397977341839, 'M0': 11.616598183948195, 'beta_max': 1.0}
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
LSD = [[[] for i in range(6)] for j in range(2)]
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

# global theory
L = np.arange(6, 3000, 1)
from model.power_spectrum import hm_ang_power_spectrum
from model.profile2D import HOD, Arnaud
g = []
g.append(HOD(nz_file="data/dndz/2MPZ_bin1.txt"))
for i in range(5):
    g.append(HOD(nz_file="data/dndz/WISC_bin%d.txt" % (i+1)))

y = Arnaud()

TT = [[[] for i in range(6)] for j in range(2)]
for i in range(6):
    TT[0][i] = hm_ang_power_spectrum(L, (g[i], g[i]), hm_correction=hm_correction, **{**kwargs[i], **cosmo_pars})
    TT[1][i] = hm_ang_power_spectrum(L, (y, g[i]), hm_correction=hm_correction, **{**kwargs[i], **cosmo_pars})


# TT0_HM = hm_ang_power_spectrum(L, (g, g), hm_correction=HaloModCorrection, **kwargs)


## FIGURE ###
import matplotlib.pyplot as plt
for i in range(6):
    # if i != 3: continue
    fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)
    plt.title("z-bin no. %d" %i)

    # gxg #
    # old data
    ax[0].plot(lsd[0][i], dd[0][i], "ro", alpha=0.2)
    # old theory
    ax[0].plot(lst[0][i], tt[0][i], "k--", alpha=0.3, lw=3, label="yxg fit")
    # new data
    ax[0].plot(LSD[0][i]["ls"], LSD[0][i]["cls"]-LSD[0][i]["nls"], "bo", alpha=0.2)
    # new theory
    ax[0].plot(L, TT[0][i], "k:", alpha=0.3, lw=3, label="new fit, old HMcorr")
    # ax[0].plot(L, TT0_HM, "k-", alpha=0.3, lw=3, label="new fit")
    ax[0].legend(loc="upper right")
    # break
    # yxg #
    # old data
    ax[1].plot(lsd[1][i], dd[1][i], "ro", alpha=0.2)
    # old theory
    ax[1].plot(lst[1][i], tt[1][i], "k--", alpha=0.3, lw=3, label="yxg fit")
    # new data
    ax[1].plot(LSD[1][i]["ls"], LSD[1][i]["cls"], "bo", alpha=0.2)
    # new theory
    ax[1].plot(L, TT[1][i], "k-", alpha=0.3, lw=3, label="new fit")

    for a in ax:
        a.loglog()
        a.legend(loc="upper right")

    fig.savefig("comparison%d.pdf" % i)
    plt.close()
